'''
    Dataset for shapenet part segmentaion.
'''

import os
import os.path
import json
import numpy as np
import sys
import torch
import torch.utils.data as data
import utils.provider as provider
import utils.orderpoints as orderpoints


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def pc_augment_to_point_num(pts, pn):
    assert(pts.shape[0] <= pn)
    cur_len = pts.shape[0]
    res = np.array(pts)
    while cur_len < pn:
        res = np.concatenate((res, pts))
        cur_len += pts.shape[0]
    return res[:pn, :]


def label_augment_to_point_num(label, pn):
    assert (label.shape[0] <= pn)
    cur_len = label.shape[0]
    res = np.array(label)
    while cur_len < pn:
        res = np.concatenate((res, label))
        cur_len += label.shape[0]
    return res[:pn]


class PartDataset(data.Dataset):
    def __init__(self, num_ptrs=2500, plane_num=32, classification=False, return_cls_label=False,
                 split='trainval', normalize=True, class_choice=None,
                 random_selection=False, random_rotation=False, random_jitter=False,
                 random_scale=False, random_translation=False, random_dropout=False, which_dir=1):
        self.random_jitter = random_jitter
        self.random_scale = random_scale
        self.random_translation = random_translation
        self.random_dropout = random_dropout
        self.random_selection = random_selection
        self.random_rotation = random_rotation

        self.plane_num = plane_num
        self.npoints = num_ptrs
        self.class_choice = class_choice
        self.which_dir = which_dir

        self.root = '/media/pwu/Data/3D_data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if self.class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.class_choice}
        else:
            self.cat = {k: v for k, v in self.cat.items()}
        print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.random_selection:
            sample_size = len(seg)
            if sample_size > self.npoints:
                sample_size = self.npoints

            choice = np.random.choice(len(seg), sample_size, replace=False)
            normal = normal[choice, :]
            seg = seg[choice]
            sampled_point_set = point_set[choice, :]
        else:
            normal = normal[0:self.npoints, :]
            seg = seg[0:self.npoints]
            sampled_point_set = point_set[0:self.npoints, :]

        if self.random_rotation:
            sampled_point_set = provider.rotate_point_cloud_instance(sampled_point_set)

        if self.random_jitter:
            sampled_point_set = provider.jitter_point_cloud_instance(sampled_point_set)

        if self.random_scale:
            sampled_point_set = provider.random_scale_point_cloud_instance(sampled_point_set)

        if self.random_translation:
            sampled_point_set = provider.random_translation_point_cloud_instance(sampled_point_set)

        if self.random_dropout:
            sampled_point_set = provider.random_point_dropout_instance_instance(sampled_point_set)

        # check if the point number is smaller than required
        ori_points_num = len(sampled_point_set)
        if ori_points_num < self.npoints:
            sampled_point_set = pc_augment_to_point_num(sampled_point_set, self.npoints)
            normal = pc_augment_to_point_num(normal, self.npoints)
            seg = label_augment_to_point_num(seg, self.npoints)

        if self.which_dir == 1:
            sampled_point_set_final = sampled_point_set
            normal_final = normal
        elif self.which_dir == 2:
            sampled_point_set_final = provider.rotate_x_dir(sampled_point_set, np.pi / 2)
            normal_final = provider.rotate_x_dir(normal, np.pi / 2)
        else:
            sampled_point_set_final = provider.rotate_z_dir(sampled_point_set, np.pi / 2)
            normal_final = provider.rotate_z_dir(normal, np.pi / 2)

        ordered_pts_final, ordered_normal_final, ordered_seg_final, quantiles_final, gather_idx, ori_point_idx = \
            orderpoints.points_grid(sampled_point_set_final, normal_final, seg, plane_num=self.plane_num)

        points_with_normals_final = np.concatenate((ordered_pts_final, ordered_normal_final), axis=1)

        points_with_normals_final = points_with_normals_final.transpose()

        return torch.from_numpy(ordered_seg_final.astype(np.int64)), \
               points_with_normals_final, np.array(quantiles_final), np.array([ori_points_num]), \
               gather_idx, ori_point_idx

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('OK')
