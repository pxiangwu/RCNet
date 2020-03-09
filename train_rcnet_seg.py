import os
import argparse
import random
import logging
from math import trunc
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from utils.part_dataset import PartDataset
from RCNetSeg import EnsembleRCNet
import time
from RCNetSeg import prepare_input_first_level
#from cosine_scheduler import CosineLRWithRestarts
import timeit


# parser = argparse.ArgumentParser()
# parser.add_argument('--gpus', default='0', help='delimited list input of GPUs', type=str)
# parser.add_argument('--dir', default='1', help='which direction', type=int)
#
#
# args = parser.parse_args()
#
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def check_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def main(args):
# ***** parameters *****
    batch_size = 8
    num_workers = 1
    num_epoch = 200
    resume_epoch = 0
    resume = False

    epoch_samples = 4421
    NUM_POINTS = 2048
    lr = 0.001
    weigh_decay = 1e-4
    milestones = [60]  # [30, 60]
    which_dir = args.dir
    OBJ_CLASS = [args.cat]


    # load data
    train_dataset = PartDataset(num_ptrs=NUM_POINTS, plane_num=32, class_choice=OBJ_CLASS,
                                random_selection=True, random_jitter=True,
                                random_scale=True, random_translation=False,
                                which_dir=which_dir,
                                split='trainval')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

    test_dataset = PartDataset(num_ptrs=NUM_POINTS, plane_num=32, class_choice=OBJ_CLASS, split='test', which_dir=which_dir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                                  shuffle=False, num_workers=num_workers)

    print('Training set size:', len(train_dataset))
    print('Test set size:', len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_classes = train_dataset.seg_classes
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    NUM_CLASS = len(seg_classes[OBJ_CLASS[0]])


    # ***** specify model and log output directory *****
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_out_dir = '/media/pwu/Data/saved_models/point_cloud/shapepart/RCNet/'
    log_out_dir = os.path.join(curr_dir, 'results')
    try:
        os.makedirs(log_out_dir)
    except OSError:
        pass

    save_model_dir_root = check_dir(os.path.join('/media/pwu/Data/saved_models/point_cloud/shapepart/RCNet/',
                                                 'save_' + str(which_dir)))
    save_model_dir_class = check_dir(os.path.join(save_model_dir_root, OBJ_CLASS[0]))
    save_model_dir = check_dir(os.path.join(save_model_dir_class, time_stamp))


    # ***** specify logger *****
    # log_dir = os.path.join(log_out_dir, 'log-' + time_stamp + '.txt')
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(message)s',
    #                     filename=log_dir,
    #                     filemode='w')
    save_weights_name = time_stamp

    # ***** build model *****
    classifier = EnsembleRCNet(device, which_dir, NUM_CLASS, NUM_POINTS)
    print(classifier)
    temp = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print("num_parameter", temp)


    # ***** load existing model *****
    model_path = os.path.join(model_out_dir, 'cls_model_' + str(resume_epoch) + '.pth')
    if model_path != '' and resume is True:
        classifier.load_state_dict(torch.load(model_path))

    # ***** define optimizer *****
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weigh_decay, amsgrad=False)
    classifier.to(device)

    # ***** scheduler *****
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # exp_lr_scheduler = CosineLRWithRestarts(optimizer, batch_size, epoch_samples, restart_period=5, t_mult=2)


    num_batch = len(train_dataset)/batch_size

    if resume:
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 0

    curr_shape_ious = None
    
    for epoch in range(start_epoch, num_epoch):
        exp_lr_scheduler.step()
        classifier.train()

        # statistic data
        single_shape_ious = []

        for b, data in enumerate(train_dataloader):
            target, points, quantiles, ori_points_num, gather_idx, ori_point_idx = data
            target = target - seg_classes[OBJ_CLASS[0]][0]
        
            target = target.to(device)
            points = points.to(device)
        
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
        
            optimizer.zero_grad()
            pred = classifier(points, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx)
        
            loss = F.cross_entropy(pred.view(-1, NUM_CLASS), target.view(-1))  # should use nll_loss, but seems like there is no difference?
            loss.backward()
            optimizer.step()
        
            # compute ious
            cur_pred_val_logits = pred.data.cpu().numpy()
            cur_pred_val = np.zeros((pred.size(0), NUM_POINTS)).astype(np.int32)
        
            ori_points_num = ori_points_num.numpy().squeeze().tolist()
            target = target.data.cpu().numpy()
            for i in range(pred.size(0)):
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, 0:ori_points_num[i]] = np.argmax(logits, 1)[0:ori_points_num[i]]
        
            for i in range(pred.size(0)):
                segp = cur_pred_val[i, 0:ori_points_num[i]]
                segl = target[i, 0:ori_points_num[i]]
                cat = OBJ_CLASS[0]
                part_ious = [0.0 for _ in range(NUM_CLASS)]
                for l in range(NUM_CLASS):
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l] = 1.0
                    else:
                        part_ious[l] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                single_shape_ious.append(np.mean(part_ious))
        
            curr_shape_ious = np.mean(single_shape_ious)
        
            msg = '[{0:d}: {1:d}/{2:d}] mean IoUs: {3:f}'.format(
                epoch, b, trunc(num_batch), curr_shape_ious)
            print(msg)
        
        curr_shape_ious = np.mean(single_shape_ious)
        msg = '*** train epoch {}, mean IoUs: {}'.format(epoch, curr_shape_ious)
        # logging.info(msg)
        print(msg)

        # evaluate
        single_shape_ious = []

        classifier.eval()
        ttime = []
        for b, data in enumerate(test_dataloader):
            target, points, quantiles, ori_points_num, gather_idx, ori_point_idx = data
            target = target - seg_classes[OBJ_CLASS[0]][0]

            target = target.to(device)
            points = points.to(device)

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
            pred = classifier(points, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx)
            stop = timeit.default_timer()
            print("time >>", stop - start)
            ttime.append(stop - start)

            # compute ious
            cur_pred_val_logits = pred.data.cpu().numpy()
            cur_pred_val = np.zeros((pred.size(0), NUM_POINTS)).astype(np.int32)

            ori_points_num = ori_points_num.numpy().squeeze().tolist()
            target = target.data.cpu().numpy()
            for i in range(pred.size(0)):
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, 0:ori_points_num[i]] = np.argmax(logits, 1)[0:ori_points_num[i]]

            for i in range(pred.size(0)):
                segp = cur_pred_val[i, 0:ori_points_num[i]]
                segl = target[i, 0:ori_points_num[i]]
                cat = OBJ_CLASS[0]
                part_ious = [0.0 for _ in range(NUM_CLASS)]
                for l in range(NUM_CLASS):
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l] = 1.0
                    else:
                        part_ious[l] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                single_shape_ious.append(np.mean(part_ious))

        curr_shape_ious = np.mean(single_shape_ious)

        msg = '*** Test mean IoUs: {0:f}'.format(curr_shape_ious)
        # logging.info(msg)
        print(msg)

        #if epoch % 10 == 0:
            # torch.save(classifier.state_dict(), '{}/{}.pth'.format(save_model_dir, curr_shape_ious))

        # logging.info(msg)

        # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (model_out_dir, epoch))

    return curr_shape_ious
