from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset.pair_dataloader import GetPairLoader
from correlativeloss import CorrelativeLoss
from models.correlative_model import CorrelativeModel
import visdom


vis = visdom.Visdom(env='correlative_finetune')
cuda = True
dset_classes_number = 19
model_ft = CorrelativeModel()
# for idx, m in enumerate(model_ft.named_modules()):
#     print(idx, '-->', m)
if cuda:
    model_ft = model_ft.cuda()
criterion = CorrelativeLoss()
# for idx, m in enumerate(criterion.named_modules()):
#     print(idx, '-->', m)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8000, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load data
batch_size = 256
image_size = 256

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#load image 
image_root = os.path.join('/1116', 'SUN')
train_list = os.path.join('/1116', 'SUN', 'train_label.txt')
test_list = os.path.join('/1116', 'SUN', 'test_label.txt')
#phase = 'train' # train or test
data_list = {
    'train':train_list,
    'test':test_list
}

dataset ={phase: GetPairLoader(
    data_root=image_root,
    phase=phase,
    data_list=data_list[phase],
    transform=data_transforms[phase])
    for phase in ['train', 'test']
}
dataset_sizes = {phase: len(dataset[phase]) for phase in ['train', 'test']}
dataloaders ={phase: torch.utils.data.DataLoader(
    dataset=dataset[phase],
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)
    for phase in ['train', 'test']}

def for_hook(module, input, output):
    for out_val in output:
        print(out_val)
    print(output)
#model_ft.feature.features[0].register_forward_hook(for_hook)
#criterion.register_backward_hook(for_hook)
# for item in model_ft.named_parameters():
#     if item[0] == 'feature.features[4].weight':
#         h = item[1].register_hook(lambda grad: print(grad))
#model_ft.feature.features[3].register_forward_hook(for_hook)
# Train and evaluate
def train_model(model, criterion, optimizer, scheduler, num_epochs=10000):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for rgb_inputs, depth_inputs, labels in dataloaders[phase]:
                rgb_inputs = rgb_inputs.to(device)
                depth_inputs = depth_inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    RGB_output, Depth_output = model(rgb_inputs, depth_inputs)
                    #RGB_output.register_hook(lambda g: print('rgboutput:\n{}'.format(g)))
                    #Depth_output.register_hook(lambda g: print('depthoutput:\n{}'.format(g)))
                    loss = criterion(RGB_output, Depth_output, labels)
                    #loss.register_hook(lambda g: print('loss:\n{}'.format(g)))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * rgb_inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([epoch_loss]), 
                 win='epoch_loss', 
                 name=phase,
                 update='append')
            vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([epoch_loss]), 
                 win='epoch_loss', 
                 name=phase,
                 update='append')

            # # deep copy the model
            # if phase == 'test' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100)
torch.save(model_ft, 'correlativemodels/bestmodel.pth')