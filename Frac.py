import torch
import math
import numpy as np
from network import objective
from utils import *

# Functions for fractional_programming algorithm
def fractional_programming(H, Pmax, var_noise):
    K = np.shape(H)[0]
    p = np.zeros(K)
    y = np.zeros(K)
    gamma = np.zeros(K)
    
    p = np.random.rand(K)*Pmax
    loss_old = 0
    for iter in range(int(1e3)):
        for i in range(K):
            gamma[i] = np.square(H[i, i]) * p[i] / (np.square(H[:, i]) @ p - np.square(H[i, i]) * p[i] + var_noise)
            y[i] = np.sqrt((1+ gamma[i])*np.square(H[i, i])*p[i]) / (np.square(H[:, i]) @ p + var_noise)
            
        for i in range(K):
            p[i] = np.minimum(Pmax, np.square(y[i])*(1+ gamma[i])*np.square(H[i,i]) / np.square(np.square(y) @ np.square(H[i, :])))

        loss = np.sum(np.log(1+gamma))
        if np.abs(loss - loss_old) <= 1e-8 and iter >= 200:
            break

        loss_old = loss
    p_opt = p
    return p_opt

def Frac(save_path = "./Frac_SNR"):
    val_path = get_dataset(path = './datasets', name = 'val.pth.tar')
    h_val_dataset = torch.load(val_path, map_location='cpu')['val_dataset']
    print('Data Generating Finished!')
    
    SNR = [0, 5, 10, 15, 20, 25, 30]
    N = 3
    cap = [None]*len(SNR)
    for i in range(len(SNR)):
        Pmax = 10**(SNR[i]/10)
        num_H = h_val_dataset.shape[0]
        var_noise = 1
        p=np.zeros((num_H, N), dtype=np.float32)
        for loop in range(num_H):
            p[loop, :] = fractional_programming(np.sqrt(h_val_dataset[loop,:,:]), Pmax, var_noise)
            if loop % int(1e4) == 0:
                print(loop / num_H, end = '  ')
        print()
        cap[i] = torch.mean(objective(torch.from_numpy(p), torch.from_numpy(h_val_dataset)))
        print('SNR ', SNR[i], 'dB: ', cap[i])
        save_checkpoint({'channel': h_val_dataset, 'power': p}, SNR[i], tag='SNR-', path = save_path)

if __name__ == "__main__":
    Frac()