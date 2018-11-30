import hdf5storage
import torch
from torchvision.transforms import Normalize, Compose
from torch.utils.data import Dataset, DataLoader


class NYUDv2Dataset(Dataset):
    def __init__(self, file_path):
        super(NYUDv2Dataset, self).__init__()
        self.file_path = file_path
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def _load_data(self):
        mat = hdf5storage.loadmat(self.file_path)
