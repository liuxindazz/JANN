from __future__ import absolute_import

import torch
from torch import nn
import numpy as np

class LiftedStructureLoss(nn.Module):
    def __init__(self, margin=1, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        """
        Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
        Implemented in `pytorch`
        """
        loss = 0
        counter = 0
        margin = self.margin
        
        bsz = inputs.size(0)
        mag = (inputs ** 2).sum(1).expand(bsz, bsz)
        sim = inputs.mm(inputs.transpose(0, 1))
        
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        
        for i in range(bsz):
            t_i = targets[i].item()
            
            for j in range(i + 1, bsz):
                t_j = targets[j].item()
                
                if t_i == t_j:
                    # Negative component
                    # !! Could do other things (like softmax that weights closer negatives)
                    l_ni = (margin - dist[i][targets != t_i]).exp().sum()
                    l_nj = (margin - dist[j][targets != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    
                    # Positive component
                    l_p  = dist[i,j]
                    
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1
        
        return loss / (2 * counter)

# def main():
#     data_size = 32
#     input_dim = 3
#     output_dim = 2
#     num_class = 4
#     # margin = 0.5
#     x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
#     # print(x)
#     w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
#     inputs = x.mm(w)
#     y_ = 8*list(range(num_class))
#     targets = Variable(torch.IntTensor(y_))

#     print(LiftedStructureLoss()(inputs, targets))


# if __name__ == '__main__':
#     main()
#     print('Congratulations to you!')
