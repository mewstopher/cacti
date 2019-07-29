from imports import *

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

