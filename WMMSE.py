import torch
import math
import numpy as np
from network import objective
from utils import *

# Functions for WMMSE algorithm
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[:, i]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    for iter in range(int(1e3)):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[i, :]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[:, i])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        if vnew - vold <= 1e-8 and iter >= 200:
            break

    p_opt = np.square(b)
    return p_opt

def WMMSE(save_path = "./WMMSE_SNR"):
    val_path = get_dataset(path = './datasets', name = 'val.pth.tar')
    h_val_dataset = torch.load(val_path, map_location='cpu')['val_dataset']
    print('Data Generating Finished!')
    
    SNR = [0, 5, 10, 15, 20, 25, 30]
    N = 3
    cap = [None]*len(SNR)
    for i in range(len(SNR)):
        Pmax = 10**(SNR[i]/10)
        num_H = h_val_dataset.shape[0]
        Pini = Pmax*np.ones(N)
        var_noise = 1
        p=np.zeros((num_H, N), dtype=np.float32)
        for loop in range(num_H):
            p[loop, :] = WMMSE_sum_rate(Pini, np.sqrt(h_val_dataset[loop,:,:]), Pmax, var_noise)
            if loop % int(1e4) == 0:
                print(loop / num_H, end = '  ')
        print()
        cap[i] = torch.mean(objective(torch.from_numpy(p), torch.from_numpy(h_val_dataset)))
        print('SNR ', SNR[i], 'dB: ', cap[i])
        save_checkpoint({'channel': h_val_dataset, 'power': p}, SNR[i], tag='SNR-', path = save_path)

if __name__ == "__main__":
    WMMSE()