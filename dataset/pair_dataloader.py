import torch.utils.data as data
from PIL import Image
import os


class GetPairLoader(data.Dataset):
    def __init__(self, data_root, data_list, phase, transform=None):
        self.root = data_root
        self.transform = transform
        self.phase = phase

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data.split()[0])
            self.img_labels.append(data.split()[1])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        rgb_imgs = Image.open(os.path.join(self.root, 'rgb', self.phase, img_paths)).convert('RGB')
        depth_imgs = Image.open(os.path.join(self.root, 'hha', self.phase, img_paths)).convert('RGB')

        if self.transform is not None:
            rgb_imgs = self.transform(rgb_imgs)
            depth_imgs = self.transform(depth_imgs)
            labels = int(labels)

        return rgb_imgs, depth_imgs, labels

    def __len__(self):
        return self.n_data
