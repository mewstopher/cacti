
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
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

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

train_size = int(0.8 * cacti_dataset.__len__())
test_size = cacti_dataset.__len__() - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cacti_dataset,
                                                            [train_size, test_size])
for i in range(len(cacti_dataset)):
    sample = cacti_dataset[i]

    print(i, sample['image'].size(), sample['cacti'].size())

    if i == 3:
        break

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False,
                                          num_workers=2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 3)
        self.conv2 = nn.Conv2d(30, 60, 3)
        self.conv3 = nn.Conv2d(60, 120, 3)
        self.fc1 = nn.Linear(1920, 50)
        self.fc2 = nn.Linear(50, 2)



    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# define loss 
error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=.0001, momentum=.9)
num_iterations = 2500
num_epochs = 2

loss_list = []
iteration_list = []
count = 0

for e in range(num_epochs):
    for i, data in enumerate(train_loader):
        batch = data['image'].to(torch.float32)
        batch = Variable(batch)
        labels = Variable(data['cacti'].view(-1))
        optimizer.zero_grad()

        outputs = net(batch)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        count += 1
        loss_list.append(loss.item())
        iteration_list.append(count)

        if count % 50 == 0:
            correct = 0
            total = 0
            for data in test_loader:
                test = Variable(data['image'].to(torch.float32))
                test_lables = Variable(data['cacti'].view(-1))
                test_outputs = net(test)
                predicted = torch.max(test_outputs.data, 1)[1]
                total += len(test_lables)
                correct += (predicted == test_lables).sum()

            accuracy = 100 * correct / float(total)
        if count % 100 == 0:
            print('iteration: {} loss: {} accuracy: {}'.format(count, loss.item(), accuracy))


