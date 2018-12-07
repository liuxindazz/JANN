#!/usr/bin/env python

"""
    pytorch_lifted_loss.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

def lifted_loss(score, target, margin=1):
    """
      Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
      Implemented in `pytorch`
    """

    loss = 0
    counter = 0
    
    bsz = score.size(0)
    mag = (score ** 2).sum(1).expand(bsz, bsz)
    sim = score.mm(score.transpose(0, 1))
    
    dist = (mag + mag.transpose(0, 1) - 2 * sim)
    dist = torch.nn.functional.relu(dist).sqrt()
    
    for i in range(bsz):
        t_i = target[i].data[0]
        
        for j in range(i + 1, bsz):
            t_j = target[j].data[0]
            
            if t_i == t_j:
                # Negative component
                # !! Could do other things (like softmax that weights closer negatives)
                l_ni = (margin - dist[i][target != t_i]).exp().sum()
                l_nj = (margin - dist[j][target != t_j]).exp().sum()
                l_n  = (l_ni + l_nj).log()
                
                # Positive component
                l_p  = dist[i,j]
                
                loss += torch.nn.functional.relu(l_n + l_p) ** 2
                counter += 1
    
    return loss / (2 * counter)

# --

if __name__ == "__main__":
    import numpy as np
    # np.random.seed(123)
    
    # score = np.random.uniform(0, 1, (20, 3))
    # target = np.random.choice(range(3), 20)
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    x = torch.tensor(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = torch.tensor(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    #y_ = 8*list(range(num_class))
    y_ = np.random.choice(num_class, data_size)
    targets = torch.tensor(torch.IntTensor(y_))
    
print(lifted_loss(torch.FloatTensor(inputs), targets))