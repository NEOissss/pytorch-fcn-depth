import os
import hdf5storage
import numpy as np
from torch import from_numpy
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset


class NYUDv2Dataset(Dataset):
    def __init__(self, dir_path, cut):
        super(NYUDv2Dataset, self).__init__()
        self.dir_path = dir_path
        self.cut = cut
        self.file_list = {'images': '/nyu_depth_v2_images.npy',
                          'depths': '/nyu_depth_v2_depths.npy',
                          'labels': '/nyu_depth_v2_labels.npy'
                          }
        self.transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.images, self.depths = self.load_data()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), from_numpy(self.depths[idx])

    def load_data(self):
        if all([os.path.exists(self.dir_path + i) for i in self.file_list.values()]):
            images = np.load(self.dir_path + self.file_list['images'])
            depths = np.load(self.dir_path + self.file_list['depths'])
        else:
            images, depths = self.prep_data()
        return images[self.cut[0]:self.cut[1]], depths[self.cut[0]:self.cut[1]]

    def prep_data(self):
        mat = hdf5storage.loadmat(self.dir_path + '/nyu_depth_v2_labeled.mat')
        images = np.array(mat['images']).transpose((3, 0, 1, 2)).copy()
        depths = np.expand_dims(np.array(mat['depths']).transpose((2, 0, 1)), 1).copy()
        labels = np.expand_dims(np.array(mat['labels']).transpose((2, 0, 1)), 1).copy()
        np.save(self.dir_path + self.file_list['images'], images)
        np.save(self.dir_path + self.file_list['depths'], depths)
        np.save(self.dir_path + self.file_list['labels'], labels)
        return images, depths


if __name__ == '__main__':
    data_opts = {'train': True, 'cut': [0, 500]}
