import os
import torch
import torch.utils.data as data
import numpy as np
from sklearn.utils import resample
from scipy import spatial

import utils.provider as provider
import timeit
import h5py
import sys
import utils.orderpoints as orderpoints
from utils.orderpoints import farthest_points_sampling


def modelnet40_load(root_dir, load_train_data=True, load_test_data=True):
    """
    Load modelnet40 data.
    """
    # ModelNet40 official train/test split
    train_files = []
    test_files = []

    if load_train_data:
        train_files = provider.getDataFiles(
            os.path.join(root_dir, 'modelnet40_ply_hdf5_2048/train_files.txt'))
    if load_test_data:
        test_files = provider.getDataFiles(
            os.path.join(root_dir, 'modelnet40_ply_hdf5_2048/test_files.txt'))

    return train_files, test_files


class ModelNet40(data.Dataset):
    def __init__(self, num_ptrs=1024, plane_num=20, random_selection=False, random_rotation=False,
                 random_jitter=False, random_scale=False, random_translation=False, random_dropout=False,
                 train=True, direction=1, root_dir='/media/pwu/Data/3D_data/'):
        self.npoints = num_ptrs
        self.plane_num = plane_num
        self.random_rotation = random_rotation
        self.random_selection = random_selection
        self.random_jitter = random_jitter
        self.random_scale = random_scale
        self.random_translation = random_translation
        self.random_dropout = random_dropout

        self.direction=direction

        if root_dir is None:
            self.root_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.root_dir = root_dir

        if train:
            data_files, _ = modelnet40_load(self.root_dir, load_train_data=True, load_test_data=False)
        else:
            _, data_files = modelnet40_load(self.root_dir, load_train_data=False, load_test_data=True)

        point_sets, labels = [], []
        for idx in range(len(data_files)):
            data_file_path = os.path.join(self.root_dir, data_files[idx])
            current_data, current_label = provider.loadDataFile(data_file_path)

            point_sets.append(current_data)
            labels.append(current_label)

        point_sets = np.concatenate(point_sets, axis=0)
        labels = np.concatenate(labels, axis=0)

        self.point_sets = point_sets
        self.labels = np.squeeze(labels)
        self.num_classes = np.unique(self.labels).size

    def __getitem__(self, index):
        selected_point_set = self.point_sets[index]

        # randomly sample npoints from the selected point cloud
        if self.random_selection:
            choice = np.random.choice(selected_point_set.shape[0], self.npoints, replace=False)
            sampled_point_set = selected_point_set[choice, :]
        else:
            sampled_point_set = selected_point_set[0:self.npoints, :]

        # random rotation for data augmentation
        if self.random_rotation:
            sampled_point_set = provider.rotate_point_cloud_instance(sampled_point_set)

        if self.random_jitter:
            sampled_point_set = provider.jitter_point_cloud_instance(sampled_point_set)
            # sampled_point_set = np.clip(sampled_point_set, -1.0, 1.0)

        if self.random_scale:
            sampled_point_set = provider.random_scale_point_cloud_instance(sampled_point_set)

        if self.random_translation:
            sampled_point_set = provider.random_translation_point_cloud_instance(sampled_point_set)

        if self.random_dropout:
            sampled_point_set = provider.random_point_dropout_instance_instance(sampled_point_set)

        if self.direction == 2:
            sampled_point_set_2 = provider.rotate_x_dir(sampled_point_set, np.pi / 2)
            ordered_pts, quantiles = orderpoints.points_grid(sampled_point_set_2, plane_num=self.plane_num)
        elif self.direction == 3:
            sampled_point_set_3 = provider.rotate_z_dir(sampled_point_set, np.pi / 2)
            ordered_pts, quantiles = orderpoints.points_grid(sampled_point_set_3, plane_num=self.plane_num)
        else:
            ordered_pts, quantiles = orderpoints.points_grid(sampled_point_set, plane_num=self.plane_num)

        label = torch.from_numpy(np.array(self.labels[index]).astype(np.int64))

        return label, ordered_pts.astype(np.float32), quantiles

    def __len__(self):
        return self.labels.shape[0]


if __name__ == '__main__':
    a = 2
