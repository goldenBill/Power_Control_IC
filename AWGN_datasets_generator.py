import os
import torch
import numpy as np

def save_dataset(dataset, path = './datasets', name = 'train.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path+'/'+name)
    torch.save(dataset, filename)
    
def generate_dataset(users = 3, train_batch = int(5e3), val_batch = int(1e4), distribution = 'AWGN', seed = 2020):
    N = users
    train_sample_num = int(train_batch)*1000
    val_sample_num = int(val_batch)*10
    
    # training dataset
    np.random.seed(seed)
    train_CH = 1/np.sqrt(2)*(np.random.randn(train_sample_num, N, N) + 1j*np.random.randn(train_sample_num, N, N))
    h_train_dataset = (np.abs(train_CH).astype(np.float32))**2
    
    # validation dataset
    val_CH = 1/np.sqrt(2)*(np.random.randn(val_sample_num, N, N) + 1j*np.random.randn(val_sample_num, N, N))
    h_val_dataset = (np.abs(val_CH).astype(np.float32))**2
    
    # save dataset
    save_dataset({'train_dataset': h_train_dataset,}, path = './datasets', name = 'train.pth.tar')
    save_dataset({'val_dataset': h_val_dataset,}, path = './datasets', name = 'val.pth.tar')
    print('Datasets are generated successfully!')
    
if __name__ == "__main__":
    generate_dataset()