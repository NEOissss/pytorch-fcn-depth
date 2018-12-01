import hdf5storage
import torch
from torchvision.transforms import Normalize
from torch.utils.data import Dataset


class NYUDv2Dataset(Dataset):
    def __init__(self, data_path, cut):
        super(NYUDv2Dataset, self).__init__()
        self.file_path = data_path
        self.cut = cut
        self.transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Load data
        mat = hdf5storage.loadmat(self.file_path)
        self.images = torch.FloatTensor(mat['images'])[self.cut[0], self.cut[1]]
        self.depths = torch.FloatTensor(mat['depths'])[self.cut[0], self.cut[1]]
        # self.labels = torch.FloatTensor(mat['labels'])[self.cut[0], self.cut[1]]

        self.images = self.images.transpose(0, 3).transpose(1, 2).transpose(2, 3)
        self.images = self.transform(self.images)

        self.depths = self.depths.transpose(0, 2).transpose(1, 2).unsqueeze(1)
        # self.labels = self.labels.transpose(0, 2).transpose(1, 2).unsqueeze(1)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[[idx]], self.depths[[idx]]


if __name__ == '__main__':
    data_opts = {'train': True, 'cut': [0, 500]}
