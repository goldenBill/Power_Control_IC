import os
import re
import torch

def save_checkpoint(state, iters, tag='', path = './models'):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path+"/{}checkpoint-{:07}.pth.tar".format(tag, iters))
    print('save in ', filename)
    torch.save(state, filename)

def get_lastest_model(path='./models', exact = ''):
    if not os.path.exists(path):
        os.mkdir(path)
    model_list = os.listdir(path+'/')
    model_list = [x for x in model_list if (exact in x and '.pth.tar' in x)]
    if model_list == []:
        return None, 0
    model_list.sort(key = lambda s: int(re.findall(r'(\d+)', s)[0]))
    lastest_model = model_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    return path + '/' + lastest_model, int(iters[0])

def get_dataset(path = './datasets', name = 'train.pth.tar'):
    if not os.path.exists(path):
        os.mkdir(path)
    dataset_list = os.listdir(path+'/')
    dataset_list = [x for x in dataset_list if (name in x)]
    if dataset_list == []:
        return None
    dataset = dataset_list[-1]
    return path + '/' + dataset