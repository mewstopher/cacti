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


class Rescale(object):
    """
    Rescale the image to a sample of a given size
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size =output_size

    def __call__(self, sample):
        image, cacti = sample['image'], sample['cacti']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'cacti': cacti}

