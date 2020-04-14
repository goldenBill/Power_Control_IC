import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from network import Approx, Lagrange, objective
from utils import *


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data
    
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.01)
        
def weights_init_to_0(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.constant_(m.weight, 0.0)
        
def get_args():
    parser = argparse.ArgumentParser("DNN_sum_rate_min")
    parser.add_argument('--train-batch-size', type=int, default=int(5e3), help='batch size')
    parser.add_argument('--val-batch-size', type=int, default=int(1e4), help='batch size')
    parser.add_argument('--total-iters', type=int, default=int(5e5), help='total iters')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='init learning rate')
    parser.add_argument('--save-path', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--show-interval', type=int, default=int(1e3), help='display interval')
    parser.add_argument('--save-interval', type=int, default=int(1e4), help='save interval')
    parser.add_argument('--power-dB', type=float, default=1, help='power constrain in dB')
    parser.add_argument('--gpu', type=str, default='0,', help='gpu_index')
    parser.add_argument('--inherit', type=str, default=None, help='pre-train')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    total_iters = int(args.total_iters)
    learning_rate = args.learning_rate
    save_path = args.save_path
    auto_continue = args.auto_continue
    show_interval = args.show_interval
    save_interval = args.save_interval
    power = 10**(args.power_dB/10)
    inherit = args.inherit
    
#     t1 = time.time()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    num_workers = 0

    # DataSetup
    N = 3

    train_path = get_dataset(path = './datasets', name = 'train.pth.tar')
    h_train_dataset = torch.from_numpy(torch.load(train_path, map_location=None if use_gpu else 'cpu')['train_dataset'])
    h_train_loader = torch.utils.data.DataLoader(
        h_train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    h_train_dataprovider = DataIterator(h_train_loader)

    val_path = get_dataset(path = './datasets', name = 'val.pth.tar')
    h_val_dataset = torch.from_numpy(torch.load(val_path, map_location=None if use_gpu else 'cpu')['val_dataset'])
    h_val_loader = torch.utils.data.DataLoader(
        h_val_dataset, batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    h_val_dataprovider = DataIterator(h_val_loader)
    print('Data Generating Finished!')

    model_Approx = Approx(inp = N**2, oup = N, hidden_dim = 10*N, P = power)
    model_Approx.apply(weights_init)
    optimizer_Approx = torch.optim.Adam(model_Approx.parameters(), lr = learning_rate)
    criterion = Lagrange()

    if use_gpu:
        model_Approx = nn.DataParallel(model_Approx)
        loss_function = criterion.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion
        device = torch.device("cpu")

    model_Approx = model_Approx.to(device)

    iters = 0
    y1 = np.zeros(total_iters)
    y2 = np.zeros(total_iters)
    if auto_continue:
        lastest_model, lastest_iters = get_lastest_model(path = save_path)
        if lastest_model is not None:
            iters = lastest_iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            y1[0:iters] = checkpoint['val_cost'][0:iters]
            y2[0:iters] = checkpoint['train_cost'][0:iters]
            model_Approx.load_state_dict(checkpoint['state_approx_dict'], strict=True)
            print('load from checkpoint with iters: ', iters)

    if iters == 0 and args.inherit != None:
        lastest_model, lastest_iters = get_lastest_model(path = args.inherit)
        if lastest_model is not None:
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model_Approx.load_state_dict(checkpoint['state_approx_dict'], strict=True)
            print('Inherit model from', args.inherit)
        
#     t2 = time.time()
#     print('prepare time: ', t2 - t1)
    while iters < total_iters:
#         t3 = time.time()
        iters += 1
        model_Approx.train()
        h_data = h_train_dataprovider.next()
        h_data = h_data.to(device)
        output_Approx = model_Approx(h_data)
        loss = loss_function(output_Approx, h_data)
        optimizer_Approx.zero_grad()
        loss.backward()
        optimizer_Approx.step()
        y2[iters - 1] = torch.mean(objective(output_Approx, h_data))
#         t4 = time.time()
#         print('train time: ', t4 - t3)

        model_Approx.eval()
        with torch.no_grad():
            h_data = h_val_dataprovider.next()
            h_data = h_data.to(device)
            approx = model_Approx(h_data)
            y1[iters - 1] = torch.mean(objective(approx, h_data))
#         print('eval time: ', time.time() - t5)

        if iters % show_interval == 0:
            print(iters, ' val: ', y1[iters - 1], ', train: ', y2[iters - 1])

        if iters % save_interval == 0:
            save_checkpoint({'state_approx_dict': model_Approx.state_dict(), 'val_cost': y1[0: iters], 'train_cost': y2[0: iters]}, iters, tag='bnps-', path = save_path)
    save_checkpoint({'state_approx_dict': model_Approx.state_dict(), 'val_cost': y1, 'train_cost': y2}, total_iters, tag='bnps-', path = save_path)

if __name__ == "__main__":
    main()