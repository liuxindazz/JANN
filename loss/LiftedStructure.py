from __future__ import absolute_import

import torch
from torch import nn
import numpy as np
import math

class LiftedStructureLoss(nn.Module):
    def __init__(self, margin=1, hard_mining=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        #self.alpha = alpha
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)

        mag = (inputs ** 2).sum(1).expand(n, n)
        sim_mat = inputs.mm(inputs.transpose(0, 1))
    
        dist_mat = (mag + mag.transpose(0, 1) - 2 * sim_mat)
        dist_mat = torch.nn.functional.relu(dist_mat).sqrt().cuda()
        # print(sim_mat)
        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = torch.tensor(torch.eye(n, n)).cuda()
        zeros_ = torch.zeros(n,n).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_dist = torch.where(pos_mask, dist_mat, zeros_)
        neg_dist = torch.where(neg_mask, dist_mat, zeros_)

        loss = 0

        len_p = pos_mask.nonzero().shape[0]
        if self.hard_mining is not None:
            for ind in torch.triu(pos_mask).nonzero():
                i = ind[0].item()
                j = ind[1].item()

                #neg_ik_component = torch.sum(torch.exp(self.margin - torch.topk(neg_dist[i][neg_dist[i].nonzero()].squeeze(),\
                #                                                             math.ceil(self.alpha*len(neg_dist[i].nonzero())))))
                #neg_jl_component = torch.sum(torch.exp(self.margin - torch.topk(neg_dist[j][neg_dist[j].nonzero()].squeeze(),\
                #                                                             math.ceil(self.alpha*len(neg_dist[j].nonzero())))))
                hardest_k, _ = torch.topk(neg_dist[i][neg_dist[i].nonzero()].squeeze(), 1, largest=False)
                neg_ik_component = torch.sum(torch.exp(self.margin - hardest_k))
                hardest_l, _ = torch.topk(neg_dist[j][neg_dist[j].nonzero()].squeeze(), 1, largest=False)
                neg_jl_component = torch.sum(torch.exp(self.margin - hardest_l))
                Jij_heat = torch.log(neg_ik_component+neg_jl_component) + pos_dist[i,j]
                loss += torch.nn.functional.relu(Jij_heat) ** 2
            return loss / len_p
        
        else:
            for ind in torch.triu(pos_mask).nonzero():
                i = ind[0].item()
                j = ind[1].item()
                neg_ik_component = torch.sum(torch.exp(self.margin - neg_dist[i][neg_dist[i].nonzero()].squeeze()))
                neg_jl_component = torch.sum(torch.exp(self.margin - neg_dist[j][neg_dist[j].nonzero()].squeeze()))
                Jij_heat = torch.log(neg_ik_component+neg_jl_component) + pos_dist[i,j]

                loss += torch.nn.functional.relu(Jij_heat) ** 2

            return loss / len_p


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    # torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123)
    # np.random.seed(123)
    x = torch.tensor(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = torch.tensor(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    #y_ = 8*list(range(num_class))
    y_ = np.random.choice(num_class, data_size)
    targets = torch.tensor(torch.IntTensor(y_))

    print(LiftedStructureLoss(hard_mining=1)(inputs, targets))
    print()


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
