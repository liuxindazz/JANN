'''import os 
from PIL import Image
import numpy as np
from torchvision import transforms
# #img_dir='/home/zzy/ai_challenger_scene_validation_20170908/scene_validation_20170908/' 
img_dir = '/1116/SUN/rgb/train/' 
img_list = os.listdir(img_dir) 
img_size = 224 
sum_r = 0 
sum_g = 0 
sum_b = 0 
count = len(img_list)
for img_name in img_list: 
    img_path = os.path.join(img_dir,img_name)  
    img = Image.open(img_path).convert('RGB')
    img = transforms.functional.resize(img,(img_size,img_size))
    img = np.array(img) 
    sum_r = sum_r+img[:,:,0].mean() 
    sum_g = sum_g+img[:,:,1].mean() 
    sum_b = sum_b+img[:,:,2].mean() 
    #count = count+1 
mean_r = sum_r/count
mean_g = sum_g/count 
mean_b = sum_b/count 

std_r = 0 
std_g = 0 
std_b = 0
for img_name in img_list: 
    img_path = os.path.join(img_dir,img_name)  
    img = Image.open(img_path).convert('RGB')
    img = transforms.functional.resize(img,(img_size,img_size))
    img = np.array(img) 
    std_r = std_r + ((img[:,:,0] - mean_r)**2).mean()
    std_g = std_g + ((img[:,:,1] - mean_g)**2).mean()
    std_b = std_b + ((img[:,:,2] - mean_b)**2).mean()
    #count = count+1 
std_r = np.sqrt(std_r/count)
std_g = np.sqrt(std_g/count)
std_b = np.sqrt(std_b/count)


mean = [mean_r/255.0, mean_g/255.0, mean_b/255.0] 
std = [std_r/255.0, std_g/255.0, std_b/255.0]
print(mean)
print(std)

'''
import sys
sys.path.append('.')
from dataset.sun_data_loader import GetLoader
import os

import torch
from torchvision import transforms
import numpy as np

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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = GetLoader(
    data_root=os.path.join(data_image_root[domain], phase),
    data_list=data_list[phase],
    transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=4845, 
                                        shuffle=False, 
                                        num_workers=0)
# for images, lalels in dataloader:
#      print(images.shape)
all_data = next(iter(dataloader))

all_samples_mean = np.mean(all_data[0].numpy(), axis=(0, 2, 3))
all_samples_std = np.std(all_data[0].numpy(), axis=(0, 2, 3))

print(all_samples_mean)
print(all_samples_std)

#RGB train
#[0.49403495 0.45738566 0.43353495]
#[0.27648422 0.28420654 0.28877014]
#RGB test
#[0.4885451  0.45687422 0.42913064]
#[0.2784015  0.28566268 0.29153702]

#hha train
#[0.7157847  0.35000023 0.45241362]
#[0.23631586 0.2384572  0.19271515]
#hha test
#[0.76849246 0.3728097  0.44725195]
#[0.22081509 0.24889638 0.19248194]