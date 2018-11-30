import hdf5storage
import torch
from torchvision.transforms import Normalize, Compose
from torch.utils.data import Dataset, DataLoader


class NYUDv2Dataset(Dataset):
    def __init__(self, file_path, cuda=True):
        super(NYUDv2Dataset, self).__init__()
        self.file_path = file_path
        self.cuda = cuda

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def _load_data(self):
        mat = hdf5storage.loadmat(self.file_path)
        self.images = torch.FloatTensor(mat['images'])
        self.depths = torch.FloatTensor(mat['depths'])
        self.labels = torch.FloatTensor(mat['labels'])

        if self.cuda:
            self.images = self.images.cuda()
            self.depths = self.depths.cuda()
            self.labels = self.labels.cuda()

