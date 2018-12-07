from __future__ import absolute_import

import torch
from torch import nn
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=1, hard_mining=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
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
        #pos_dist = torch.triu(pos_dist)

        loss = 0
        len_p = pos_mask.nonzero().shape[0]
        for ind in torch.triu(pos_mask).nonzero():
            i = ind[0].item()
            j = ind[1].item()
            #if (i<j):
            neg_ik_component = torch.sum(torch.exp(self.margin - neg_dist[i][neg_dist[i].nonzero()].squeeze()))
            neg_jl_component = torch.sum(torch.exp(self.margin - neg_dist[j][neg_dist[j].nonzero()].squeeze()))
            Jij_heat = torch.log(neg_ik_component+neg_jl_component) + pos_dist[i,j]

            loss += torch.nn.functional.relu(Jij_heat) ** 2

        return loss / len_p


'''
        for i, pos_pair_ in enumerate(pos_sim):
            # print(i)
            pos_pair_ = pos_pair_[pos_pair_.nonzero()].squeeze()
            neg_pair_ = neg_sim[i]
            neg_pair_ = neg_pair_[neg_pair_.nonzero()].squeeze()
            
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:
                # print(scale_value)
                # print(self.alpha,  self.beta, self.margin)
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
                # pos_pair = pos_pair[1:]
                
                # if len(pos_pair) > 1:
                #     pos_pair = torch.masked_select(pos_pair, pos_pair > torch.mean(neg_pair_))
                # if len(neg_pair) > 1:
                #     neg_pair = torch.masked_select(neg_pair, neg_pair > pos_pair_[-1] - 0.05)  
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue
                pos_loss =  torch.log(torch.sum(torch.exp(-(pos_pair - base))))
                neg_loss =  torch.log(torch.sum(torch.exp( (neg_pair - base))))
                # pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta*(pos_pair - self.margin))))
                # neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha*(neg_pair - self.margin))))
                loss.append(pos_loss + neg_loss)

            else:
                # print('no-Hard mining')
                neg_pair = neg_pair_
                pos_pair = pos_pair_
                pos_loss = torch.sum(pos_pair)
                neg_loss = torch.log(torch.sum(torch.exp((neg_pair - self.margin))))
                loss.append(pos_loss + neg_loss)


        loss = torch.sum(torch.cat(loss))/n
        prec = float(c)/n
        neg_d = torch.mean(neg_sim).data[0]
        pos_d = torch.mean(pos_sim).data[0]

        return loss, prec, pos_d, neg_d'''


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

    print(LiftedStructureLoss()(inputs, targets))
    print()


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
