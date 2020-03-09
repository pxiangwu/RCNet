import h5py
import os
from pyntcloud import PyntCloud
import numpy as np
from sampling import farthest_points_sampling
import csv
import logging
# from show3d_balls import showpoints


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


lines = [
    [[0, 0, 0], [0, 0, 1]],
    [[1, 0, 0], [1, 0, 1], [1, 1, 1]],
]

log_dir = 'log_pc.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=log_dir,
                    filemode='w')

sampled_points_num = 10000
fps_points_num = 2048
batch_size = 3000

data_path = '/media/pwu/Data/3D_data/ShapeNetCore/ShapeNetCore.v2'
csv_file = '/media/pwu/Data/3D_data/ShapeNetCore/all.csv'

# -- process CSV
csv_fid = open(csv_file, 'rt')
reader = csv.reader(csv_fid)

csv_data = []
for row in reader:
    csv_data.append(row)

csv_data = np.array(csv_data[1:])

classes = np.unique(csv_data[:, 1])
classes_map = {classes[i]: i for i in range(len(classes))}

# -- process mesh
dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
dirs.sort()

train_pc_data = []
train_pc_label = []

val_pc_data = []
val_pc_label = []

total_cnt = 0
train_thresh_cnt = 0
val_thresh_cnt = 0

train_cnt = 0
val_cnt = 0

train_saved_file_cnt = 0
val_saved_file_cnt = 0

for dir_name in dirs:
    curr_dir = os.path.join(data_path, dir_name)
    sub_dirs = [d for d in os.listdir(curr_dir) if os.path.isdir(os.path.join(curr_dir, d))]
    sub_dirs.sort()

    for sub_dir_name in sub_dirs:  # sub_dir_name: the hashing code
        file_dir = os.path.join(curr_dir, sub_dir_name, 'models')
        total_cnt += 1

        # find the label
        idx = np.where(csv_data[:, 3] == sub_dir_name)[0]
        if len(idx) == 0:
            continue

        label_code = csv_data[idx, 1]
        label = classes_map[label_code[0]]

        # find the split
        split = csv_data[idx, 4]

        # if is test file, skip
        if split == 'test':
            continue

        # find the .off file
        file_name = [f for f in os.listdir(file_dir) if f.lower().endswith('.off')]
        if len(file_name) != 1:
            logging.info('Oops!' + file_dir)
            continue

        file_path = os.path.join(file_dir, file_name[0])

        mesh = PyntCloud.from_file(file_path)

        point_cloud = mesh.get_sample("mesh_random", n=sampled_points_num, rgb=False, normals=False)
        point_cloud = point_cloud.values

        pc = pc_normalize(point_cloud)
        pc = farthest_points_sampling(pc, fps_points_num)

        # gather data
        if split == 'val':
            val_pc_data.append(pc)
            val_pc_label.append(label)

            val_cnt += 1
            val_thresh_cnt += 1

            if val_thresh_cnt == batch_size:
                saved_file = os.path.join(data_path, 'val_' + str(val_saved_file_cnt) + '.hdf5')
                fid = h5py.File(saved_file, 'w')

                val_pc_data = np.array(val_pc_data)
                val_pc_label = np.array(val_pc_label)

                fid.create_dataset('data', data=val_pc_data)
                fid.create_dataset('label', data=val_pc_label)
                fid.close()

                # reset
                val_saved_file_cnt += 1
                val_thresh_cnt = 0
                val_pc_data = []
                val_pc_label = []

        elif split == 'train':
            train_pc_data.append(pc)
            train_pc_label.append(label)

            train_cnt += 1
            train_thresh_cnt += 1

            if train_thresh_cnt == batch_size:
                saved_file = os.path.join(data_path, 'train_' + str(train_saved_file_cnt) + '.hdf5')
                fid = h5py.File(saved_file, 'w')

                train_pc_data = np.array(train_pc_data)
                train_pc_label = np.array(train_pc_label)

                fid.create_dataset('data', data=train_pc_data)
                fid.create_dataset('label', data=train_pc_label)
                fid.close()

                # reset
                train_saved_file_cnt += 1
                train_thresh_cnt = 0
                train_pc_data = []
                train_pc_label = []

        print('Total cnt:', total_cnt, 'train_cnt:', train_cnt, 'train_thresh_cnt:', train_thresh_cnt, 'val_cnt:', val_cnt, 'val_thresh_cnt', val_thresh_cnt)

# store the remaining
# - val data
saved_file = os.path.join(data_path, 'val_' + str(val_saved_file_cnt) + '.hdf5')
fid = h5py.File(saved_file, 'w')

val_pc_data = np.array(val_pc_data)
val_pc_label = np.array(val_pc_label)

fid.create_dataset('data', data=val_pc_data)
fid.create_dataset('label', data=val_pc_label)
fid.close()

# - train data
saved_file = os.path.join(data_path, 'train_' + str(train_saved_file_cnt) + '.hdf5')
fid = h5py.File(saved_file, 'w')

train_pc_data = np.array(train_pc_data)
train_pc_label = np.array(train_pc_label)

fid.create_dataset('data', data=train_pc_data)
fid.create_dataset('label', data=train_pc_label)
fid.close()
