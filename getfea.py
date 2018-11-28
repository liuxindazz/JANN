from models.sun_model import CNNModel
from PIL import Image
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from dataset.sun_data_loader import GetLoader
import numpy as np

batch_size = 128
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
phase = 'train' # train or test
if phase=='train':
    data_list = train_list
else:
    data_list = test_list

domain = 'source' #source or target
if domain == 'source':
    data_image_root = source_image_root
    feaname = 'rgb_'+phase+'_features.npy'
else:
    data_image_root = target_image_root
    feaname = 'hha_'+phase+'_features.npy'

model_name = 'sun_model_epoch_42.pth'

print(data_list)
dataset_source = GetLoader(
    data_root=os.path.join(data_image_root, phase),
    data_list=data_list,
    transform=data_transforms['test']
)
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)
data_source_iter = iter(dataloader_source)

# load model
model_root = os.path.join('models', model_name)
my_net = torch.load(model_root)
for idx, m in enumerate(my_net.named_modules()):
    print(idx, '-->', m)

# inter_feature = {}
# def make_hook(name, flag):
#      if flag == 'forward':
#          def hook(my_net, input, output):
#              inter_feature[name] = input
#          return hook
#      elif flag == 'backward':
#          def hook(my_net, input, output):
#              inter_gradient[name] = output
#          return hook
#      else:
#          assert False
features_4096 = []
def for_hook(module, input, output):
    #print(module)
    #print(output)
    for out_val in output:
        features_4096.append(out_val.data.cpu().numpy().tolist())
    #print(len(features_4096))    
    # for val in input:
    #     print("input val:",val.shape)
    # for out_val in output:
    #     print("output val:", out_val)

#k = my_net.feature.classifier.register_forward_hook(make_hook('feature', 'forward'))
my_net.class_classifier.c_fc1.register_forward_hook(for_hook)
i = 0
while 1:
    try:
        data_source = data_source_iter.next()
        s_img, s_label = data_source
    except StopIteration:
        #data_source_iter = iter(dataloader_source)
        #s_img, s_label = data_source_iter.next()
        break
    input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
    input_img.resize_as_(s_img).copy_(s_img)
    inputv_img = Variable(input_img).cuda()

    _ = my_net(inputv_img, 1)
print(np.array(features_4096).shape)
np.save(feaname, features_4096)