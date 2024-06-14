import io
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'dgcnn-master/pytorch'))

import torch
import argparse
import numpy as np
import torch.nn.functional as F
from data import ModelNet40, download
from torch.utils.data import DataLoader, Subset
from torch import nn
from mydgcnn import DGCNN_CAMGRAD
from util import cal_loss
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pc_util
from cprint import *
from main import drop_points
from sklearn.preprocessing import QuantileTransformer
import math

def load_model():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=3072, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--num_drops', type=int, default=10, help='num of points to drop each step')
    parser.add_argument('--num_steps', type=int, default=10, help='num of steps to drop each step')
    parser.add_argument('--drop_neg', action='store_true',help='drop negative points')
    parser.add_argument('--power', type=int, default=6, help='x: -dL/dr*r^x')
    parser.add_argument('--subset', type=bool, default=False, help='If dataset has to be subesetted or not')
    parser.add_argument("-f", "--file", required=False)
    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DGCNN_CAMGRAD(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('dgcnn-master/pytorch/pretrained/model.1024.t7', device))
    model = model.eval()
    return device, args, model

def calc_saliency(cloud, label, args, model, device):
    data = torch.from_numpy(cloud).unsqueeze(0)
    label = torch.tensor(label, dtype=torch.int64).unsqueeze(0)
    data, label = data.to(device), label.to(device).squeeze()
    data = data.permute(0, 2, 1)
    data.requires_grad_()
    adv, saliency = drop_points(data, label, args, model, device)
    return adv, saliency

def normalize_log(array):
    offset = array.min() - 1e-8
    colors = array - offset
    colors = np.log(colors)
    mean = np.mean(colors)
    std = np.std(colors)
    colors = (colors - mean)/std
    colors = (colors - colors.min()) / (colors.max() - colors.min()) * 1000
    return colors

def normalize_quantile(array):
    colors = array.reshape(-1, 1)
    quantile_transformer = QuantileTransformer(output_distribution='uniform')
    colors = quantile_transformer.fit_transform(colors)
    colors = colors.reshape(array.shape)
    colors = (colors - colors.min()) / (colors.max() - colors.min()) * 1000
    return colors

def normalize_regions(array, percentile=20):
    sorted = np.sort(array)
    split = math.floor(array.size*100/percentile)
    red = sorted[:, -split]
    green = sorted[:, split]
    colors = np.full_like(array, 500)
    colors[array >= red] = 1000
    colors[array <= green] = 0