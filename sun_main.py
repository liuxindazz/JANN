from __future__ import print_function
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.sun_model import CNNModel
import numpy as np
from sun_test import test
import visdom
from dataset.sun_data_loader import GetLoader

#cuda = False

vis = visdom.Visdom(env='dann_sun_rgbd')
source_dataset_name = 'rgb'
target_dataset_name = 'hha'
source_image_root = os.path.join('/1116', 'SUN', source_dataset_name)
target_image_root = os.path.join('/1116', 'SUN', target_dataset_name)
model_root = os.path.join('models')
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 256
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_list = os.path.join('/1116', 'SUN', 'train_label.txt')
dataset_source = GetLoader(
    data_root=os.path.join(source_image_root, 'train'),
    data_list=train_list,
    transform=data_transforms['train']
)
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'train'),
    data_list=train_list,
    transform=data_transforms['train']
)
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

# for k in dataloader_target:
#     print(k[0].size())

# load model

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training

for epoch in xrange(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        try:
            data_source = data_source_iter.next()
            s_img, s_label = data_source
        except StopIteration:
            data_source_iter = iter(dataloader_source)
            s_img, s_label = data_source_iter.next()
        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        class_output, domain_output = my_net(input_data=inputv_img, alpha=alpha)
        err_s_label = loss_class(class_output, classv_label)
        err_s_domain = loss_domain(domain_output, domainv_label)

        # training model using target data
        try:
            data_target = data_target_iter.next()
            t_img, t_label = data_target
        except StopIteration:
            data_target_iter = iter(dataloader_source)
            t_img, t_label = data_target_iter.next()
        my_net.zero_grad()
        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        class_output, domain_output = my_net(input_data=inputv_img, alpha=alpha)
        err_t_label = loss_class(class_output, classv_label)
        err_t_domain = loss_domain(domain_output, domainv_label)
        err = err_t_label + err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

        print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                 err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))
        vis.line(X=torch.FloatTensor([epoch*len_dataloader+i+1]), 
                 Y=torch.FloatTensor([err_s_label]), 
                 win='err_s_label', 
                 opts=dict(title='err_s_label'),
                 update='append' if i> 0 else None)
        vis.line(X=torch.FloatTensor([epoch*len_dataloader+i+1]), 
                 Y=torch.FloatTensor([err_s_domain]), 
                 win='err_s_domain', 
                 opts=dict(title='err_s_domain'),
                 update='append' if i> 0 else None)
        vis.line(X=torch.FloatTensor([epoch*len_dataloader+i+1]), 
                 Y=torch.FloatTensor([err_t_domain]), 
                 win='err_t_domain',
                 opts=dict(title='err_t_domain'),
                 update='append' if i> 0 else None) 
        vis.line(X=torch.FloatTensor([epoch*len_dataloader+i+1]), 
                 Y=torch.FloatTensor([err_t_label]), 
                 win='err_t_label',
                 opts=dict(title='err_t_label'),
                 update='append' if i> 0 else None)         
    torch.save(my_net, '{0}/sun_model_epoch_{1}.pth'.format(model_root, epoch))
    test(source_dataset_name, epoch)
    test(target_dataset_name, epoch)

print('done')
