import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class DigitsDataset(Dataset):
    
    def __init__(self, base_path, subset_name, dataidxs=None, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(base_path+'digits5/'+'{}_train.pkl'.format(subset_name), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(base_path+'digits5/'+'{}_test.pkl'.format(subset_name), allow_pickle=True)

        label_dict={'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}

        self.dataidxs = dataidxs

        self.labels = [label_dict[text] for text in self.text_labels] # transfer str to num
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

        if self.dataidxs is not None:
            self.paths = [self.paths[i] for i in self.dataidxs]
            self.labels = [self.labels[i] for i in self.dataidxs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, 'digits5/', self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
  

class DomainNetDataset(Dataset):
    
    def __init__(self, base_path, subset_name, dataidxs=None, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(os.path.join(base_path, 'DomainNet', 'pkls', '{}_train.pkl'.format(subset_name)), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(os.path.join(base_path, 'DomainNet', 'pkls', '{}_test.pkl'.format(subset_name)), allow_pickle=True)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                    'windmill': 7, 'wine_glass': 8, 'zebra': 9}
        
        self.dataidxs = dataidxs

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

        if dataidxs is not None:
            self.paths = [self.paths[i] for i in dataidxs]
            self.labels = [self.labels[i] for i in dataidxs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    

class OfficeDataset(Dataset):
    
    def __init__(self, base_path, subset_name, dataidxs=None, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(base_path+'office10/'+'{}_train.pkl'.format(subset_name), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(base_path+'office10/'+'{}_test.pkl'.format(subset_name), allow_pickle=True)

        for i in range(len(self.paths)):
            tmp = self.paths[i].split('/')[1:]
            self.paths[i] = '/'.join(tmp)

        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}

        self.dataidxs = dataidxs

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

        if dataidxs is not None:
            self.paths = [self.paths[i] for i in dataidxs]
            self.labels = [self.labels[i] for i in dataidxs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, 'office10/', self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label