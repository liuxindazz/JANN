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
from dataset.sun_data_loader import GetLoader
import visdom


vis = visdom.Visdom(env='dann_finetune')
cuda = True
dset_classes_number = 19
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft = models.alexnet(pretrained=True) 
# model_ft.classifier  = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, dset_classes_number),
# )

model_ft = models.__dict__['alexnet'](num_classes=365)
checkpoint = torch.load('/1116/models/alexnet_places365.pth.tar', map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model_ft.load_state_dict(state_dict)
new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-1])
new_classifier.add_module('fc9', nn.Linear(4096, dset_classes_number))
model_ft.classifier = new_classifier

for idx, m in enumerate(model_ft.named_modules()):
    print(idx, '-->', m)

if cuda:
    model_ft = model_ft.cuda()
class_weight = np.load('sun_class_weights.npy')
class_weight = torch.from_numpy(class_weight).float().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8000, gamma=0.1)

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
source_image_root = os.path.join('/1116', 'SUN', 'rgb')
target_image_root = os.path.join('/1116', 'SUN', 'hha')
train_list = os.path.join('/1116', 'SUN', 'train_label.txt')
test_list = os.path.join('/1116', 'SUN', 'test_label.txt')
#phase = 'train' # train or test
data_list = {
    'train':train_list,
    'test':test_list
}

domain = 'source' #source or target
data_image_root = {'source':source_image_root, 
                    'traget':target_image_root
}

dataset ={phase: GetLoader(
    data_root=os.path.join(data_image_root[domain], phase),
    data_list=data_list[phase],
    transform=data_transforms[phase])
    for phase in ['train', 'test']
}
dataset_sizes = {phase: len(dataset[phase]) for phase in ['train', 'test']}
dataloaders ={phase: torch.utils.data.DataLoader(
    dataset=dataset[phase],
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)
    for phase in ['train', 'test']}

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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([epoch_loss]), 
                 win='epoch_loss', 
                 name=phase,
                 update='append')
            vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([epoch_acc]), 
                 win='epoch_acc', 
                 name=phase,
                 update='append')
            vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([epoch_loss]), 
                 win='epoch_loss', 
                 name=phase,
                 update='append')
            vis.line(X=torch.FloatTensor([epoch+1]), 
                 Y=torch.FloatTensor([epoch_acc]), 
                 win='epoch_acc', 
                 name=phase,
                 update='append')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10000)
torch.save(model_ft, 'models/bestmodel.pth')