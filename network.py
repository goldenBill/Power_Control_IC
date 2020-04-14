import torch
import torch.nn as nn
import numpy as np

class ReLU_threshold(nn.Module):

    def __init__(self, threshold):
        super(ReLU_threshold, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        return torch.clamp(inputs, 0, self.threshold)

class Approx(nn.Module):
    def __init__(self, inp=2**2, oup=2, hidden_dim=20, P = 1):
        super(Approx, self).__init__()
        self.inp = inp
        self.oup = oup
        self.hidden_dim = hidden_dim
        self.P = P
        self.architecture = nn.Sequential(
            nn.Linear(self.inp, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.oup, bias = True),
            nn.BatchNorm1d(self.oup),
            ReLU_threshold(threshold = self.P))
        
    def forward(self, h):
        N = h.shape[0]
        return self.architecture(h.contiguous().view(N, -1))
    
class Lagrange(nn.Module):
    def __init__(self):
        super(Lagrange, self).__init__()
        
    def forward(self, approx, h):
        result = -objective(approx, h)
        return torch.mean(result)
    
def objective(p, h):
    N, W = p.shape
    h_gain = p.view(N, W, -1)*h
    denominator = (h_gain - torch.diag_embed(h_gain.diagonal(dim1 = 1, dim2 = 2), dim1 = 1, dim2 = 2)).sum(dim = 1) + 1
    numerator = torch.diagonal(h_gain, dim1 = 1, dim2 = 2)
    return torch.sum(torch.log(1 + numerator / denominator), dim = 1)
    
    
if __name__ == "__main__":
    model_Approx = Approx(inp = 2**2, oup = 2, hidden_dim = 10*2, P = 1)
    
    h_data = torch.zeros(10, 2, 2)
    h_data.exponential_(lambd=1)
    output_Approx = model_Approx(h_data)
    print(output_Approx.size())