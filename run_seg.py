from train_rcnet_seg import main
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0', help='delimited list input of GPUs', type=str)
parser.add_argument('--cat', default='Guitar', help='the shape category', type=str)
parser.add_argument('--dir', default='1', help='which direction', type=int)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

iou = main(args)
print("mIou: {}".format(miou))
