from __future__ import absolute_import

import torch
from torch import nn
def print_grad(grad):
    print('************')
    print(grad)

inputs_x = torch.randn(10, 20, requires_grad=True)
inputs_z = torch.randn(10, 20, requires_grad=True)
targets = torch.tensor([1,1,1,2,2,2,3,3,3,2])

if(inputs_x.size(0)==inputs_z.size(0) and inputs_x.size(0) == targets.size(0)):
    pass
else:
    print('length are different')
    import sys
    try:
        sys.exit(0)
    except:
        print 'die'
    finally:
        print 'cleanup'

loss = 0
counter = 0
margin = 1

bsz = inputs_x.size(0)
mag_x = (inputs_x ** 2).sum(1).expand(bsz, bsz)
mag_z = (inputs_z ** 2).sum(1).expand(bsz, bsz)
sim = inputs_x.mm(inputs_z.transpose(0, 1))

dist = (mag_x + mag_z.transpose(0, 1) - 2 * sim + 1e-7)
dist = torch.nn.functional.relu(dist).sqrt()
sim.register_hook(print_grad)
dist.register_hook(print_grad)

for i in range(bsz):
    t_i = targets[i].item()
    
    for j in range(i + 1, bsz):
        t_j = targets[j].item()
        
        if t_i == t_j:
            # Positive component
            l_p = dist[i,j]
            # Negative component
            # !! Could do other things (like softmax that weights closer negatives)
            #R xi n_zj
            l_ni = (margin - dist[i][targets != t_i]).exp().sum().log()
            l_term1 = torch.nn.functional.relu(l_ni + l_p) ** 2
            #_ = l_term1.register_hook(lambda grad: print(grad))

            #print(targets != t_j)
            l_nj = (margin - dist.t()[j][targets != t_j]).exp().sum().log()
            l_term2 = torch.nn.functional.relu(l_nj + l_p) ** 2
            #l_term2.register_hook(lambda g: print('l_term2:\n{}'.format(g)))
            loss += l_term1 +l_term2
            counter += 1

print(counter)
loss = loss / (4 * counter)
loss.backward()



# x = torch.randn(2, 1, requires_grad=True)
# y = x + 2
# z = torch.mean(torch.pow(y, 2))
# y.register_hook(print_grad)
# z.backward()
