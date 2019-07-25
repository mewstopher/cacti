
# this is a tutorial for loading data with pytorch. later we will make this into
# several scripts
# lets start with loading in the train.csv file that contains names of pictures and labels

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()

# dataset has 17500 images (rows) and labels 0, 1 for no-cacti, cacti


class CactiDataset(Dataset):
    """ cacti dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        self.cacti_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        returns the lenth of csv file read in
        """
        return len(self.cacti_frame)

    def __getitem__(self, idx):
        """
        reads in images. this is mememory efficient
        as it will only read in images as required,
        as opposed to all at once

        PARAMS
        -----------------------
        idx: index
        """
        img_name = os.path.join(self.root_dir, self.cacti_frame.iloc[idx, 0])
        image = io.imread(img_name)
        cacti = self.cacti_frame.iloc[idx, 1]
        sample = {'image': image, 'cacti': cacti}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Converts numpy array to tensor
    swap numpy array HxWxC -> tensor of
    shape CxHxW
    """
    def __call__(self, sample):
        image, cacti = sample['image'], np.array([sample['cacti']])
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'cacti': torch.from_numpy(cacti)}

cacti_dataset = CactiDataset(csv_file="../input/train.csv", root_dir="../input/train/",
                             transform=transforms.Compose([ToTensor()]))

for i in range(len(cacti_dataset)):
    sample = cacti_dataset[i]

    print(i, sample['image'].size(), sample['cacti'].size())

    if i == 3:
        break

data_loader = torch.utils.data.DataLoader(sample, batch_size=4, shuffle=False, num_workers=2)

for i, data in enumerate(data_loader):
    images, dat = data
    images.shape
