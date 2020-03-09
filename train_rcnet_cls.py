import argparse
import os
import random
import time
import logging
from math import trunc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from utils.datasets import ModelNet40
from RCNetCls import EnsembleRCNet, prepare_input_first_level
import timeit
#from adamw import AdamW
#from cosine_scheduler import CosineLRWithRestarts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def blue(x):
    return '\033[94m' + x + '\033[0m'


# *****************************************************
# parameters
which_dir = 1
batch_size = 32
num_workers = 2
num_epoch = 200
resume_epoch = 0
resume = False
lr = 0.001
milestone = [30]  # [30, 60]
#milestone = [30, 60]  # [30, 60]
lr_decay = 0.1
# *****************************************************

# load data
train_dataset = ModelNet40(num_ptrs=1024, plane_num=32, direction=which_dir,
                           random_selection=True, random_jitter=True,
                           random_scale=True, random_translation=True,
                           train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

test_dataset = ModelNet40(num_ptrs=1024, plane_num=32, direction=which_dir,
                          train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False, num_workers=num_workers)
num_classes = train_dataset.num_classes

print('Training set size:', len(train_dataset))
print('Test set size:', len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# *****************************************************
# *****************************************************
# specify model and log output directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
log_out_dir = os.path.join(curr_dir, 'results')
model_out_dir = '/media/pwu/Data/saved_models/point_cloud/modelnet40_cls/RCNet'
try:
    os.makedirs(log_out_dir)
except OSError:
    pass

# specify logger
time_stamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(log_out_dir, 'log-' + time_stamp + '.txt')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=log_dir,
                    filemode='w')

save_weights_name = time_stamp
# *****************************************************
# *****************************************************
# build model
# classifier = EnsembleRCNet(which_dir=which_dir, device=device, k=num_classes)
classifier = EnsembleRCNet(which_dir=which_dir, device=device, k=num_classes)
print(classifier)

# load existing model
model_path = os.path.join(model_out_dir, 'cls_model_' + str(resume_epoch) + '.pth')
if model_path != '' and resume is True:
    classifier.load_state_dict(torch.load(model_path))

# define optimizer


# optimizer = optim.Adam(classifier.RCNet1.rnn.parameters(), lr=lr, weight_decay=1e-4)
# all_optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
# optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4, amsgrad=False)
classifier.to(device)

# scheduler
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=lr_decay)
# rnn_exp_lr_scheduler = lr_scheduler.MultiStepLR(rnn_optimizer, milestones=milestone, gamma=lr_decay)
# all_exp_lr_scheduler = lr_scheduler.MultiStepLR(all_optimizer, milestones=milestone, gamma=lr_decay)
# exp_lr_scheduler = CosineLRWithRestarts(optimizer, batch_size, 9840, restart_period=10, t_mult=2)
# *****************************************************
# *****************************************************
num_batch = len(train_dataset)/batch_size

if resume:
    start_epoch = resume_epoch + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, num_epoch):
    train_correct = 0
    train_loss = 0
    test_correct = 0
    test_loss = 0
    num_train_data = 0
    num_test_data = 0

    # rnn_exp_lr_scheduler.step()
    # all_exp_lr_scheduler.step()
    exp_lr_scheduler.step()
    classifier.train()

    for b, data in enumerate(train_dataloader):
        target, points, quantiles = data
        target = target.to(device)

        # ***************************************************************
        # first, prepare the input to rnn
        seq_data, seq_len, inverse_index = prepare_input_first_level(points, quantiles)
        seq_data = torch.from_numpy(seq_data.astype(np.float32))
        seq_data = seq_data.to(device)

        # next, prepare for the data index for convolution
        batch_num = quantiles.shape[0]
        plane_num = quantiles.shape[1]
        items_indices = np.array([], dtype=np.int32)
        cnt = 0
        for i in range(batch_num):
            plane_slice = []
            for j in range(plane_num):
                item = []
                for k in range(plane_num):
                    num = quantiles[i, j, k]
                    if num != 0:
                        items_indices = np.append(items_indices, cnt)
                    cnt = cnt + 1
        # ***************************************************************

        # rnn_optimizer.zero_grad()
        # all_optimizer.zero_grad()
        optimizer.zero_grad()

        pred = classifier(points, quantiles, seq_data, seq_len, inverse_index, items_indices)
        loss = F.cross_entropy(pred, target)  # should use nll_loss, but didn't find difference on the results...
        loss.backward()
        # all_optimizer.step()
        optimizer.step()

        # rnn_optimizer.step()

        pred_choice = pred.data.max(1)[1]
        train_correct = train_correct + pred_choice.eq(target.data).cpu().sum().numpy()
        train_loss = train_loss + loss.data.item()
        num_train_data = num_train_data + target.size()[0]

        msg = '[{0:d}: {1:d}/{2:d}] accuracy: {3:f}'.format(
            epoch, b, trunc(num_batch), train_correct / float(num_train_data))
        print(msg)

    # evaluate
    classifier.eval()
    ttime = []
    for b, data in enumerate(test_dataloader):
        target, points, quantiles = data
        target = target.to(device)

        # ***************************************************************
        # first, prepare the input to rnn
        seq_data, seq_len, inverse_index = prepare_input_first_level(points, quantiles)
        seq_data = torch.from_numpy(seq_data.astype(np.float32))
        seq_data = seq_data.to(device)

        # next, prepare for the data index for convolution
        batch_num = quantiles.shape[0]
        plane_num = quantiles.shape[1]
        items_indices = np.array([], dtype=np.int32)
        cnt = 0
        for i in range(batch_num):
            plane_slice = []
            for j in range(plane_num):
                item = []
                for k in range(plane_num):
                    num = quantiles[i, j, k]
                    if num != 0:
                        items_indices = np.append(items_indices, cnt)
                    cnt = cnt + 1
        # ***************************************************************

        start = timeit.default_timer()
        pred = classifier(points, quantiles, seq_data, seq_len, inverse_index, items_indices)
        stop = timeit.default_timer()
        print("time", stop - start)
        ttime.append(stop - start)

        if b % 20 == 0:
            print("avg time", np.average(ttime))
            ttime = []

        pred_choice = pred.data.max(1)[1]
        test_correct = test_correct + pred_choice.eq(target.data).cpu().sum().numpy()
        num_test_data = num_test_data + target.size()[0]

    curr_accuracy = test_correct / float(num_test_data)

    if epoch % 10 == 0:
        torch.save(classifier.state_dict(), '%s/epoch_%d_%d_%s.pth' % (model_out_dir, epoch, which_dir, save_weights_name))

    msg = '*** Test accuracy: {}'.format(curr_accuracy)
    logging.info(msg)
    print(msg)
