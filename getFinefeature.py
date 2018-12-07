import os
import torch
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

model_root = os.path.join('/1116/tmp/models', 'myprefix_mymodel_215.pth')
model_ft = torch.load(model_root)
for idx, m in enumerate(model_ft.named_modules()):
    print(idx, '-->', m)

#load data
cuda = True
if cuda:
    model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load data
batch_size = 128
image_size = 256

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.70183206, 0.33803278, 0.4370506 ], [0.23395756, 0.2259527, 0.18689048])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.70183206, 0.33803278, 0.4370506 ], [0.23395756, 0.2259527, 0.18689048])
    ]),
}

#load image 
source_image_root = os.path.join('/1116', 'SUN', 'rgb')
target_image_root = os.path.join('/1116', 'SUN', 'hha')
train_list = os.path.join('/1116', 'SUN', 'train_label.txt')
test_list = os.path.join('/1116', 'SUN', 'test_label.txt')
phase = 'test' # train or test
data_list = {
    'train':train_list,
    'test':test_list
}

domain = 'traget' #source or target
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

features_4096 = []
def for_hook(module, input, output):
    print(module)
    #print(output)
    for out_val in input[0]:
        features_4096.append(out_val.data.cpu().numpy().tolist())
    print(len(features_4096))

model_ft.classifier[6].register_forward_hook(for_hook)

pred_label = []
for input_img, _ in dataloaders[phase]:
    input_img = input_img.to(device)

    labels = model_ft(input_img)
    for label in labels:
        _, preds = torch.max(label, 0)
        pred_label.append(preds.data.cpu().tolist())
print(np.array(features_4096).shape)
if domain == 'source':
    feaname = 'rgb_'+phase+'_features.npy'
else:
    feaname = 'hha_'+phase+'_features.npy'
np.save(feaname, features_4096)
np.save('pred_label.npy', pred_label)