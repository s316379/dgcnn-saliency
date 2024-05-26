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
from torch.utils.data import DataLoader
from torch import nn
from mydgcnn import DGCNN_CAMGRAD
from util import cal_loss
from utils import render_cloud
import matplotlib.pyplot as plt


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    # download modelnet if not exists
    download()
    test_loader = DataLoader(ModelNet40(partition='test', num_points=1024),
                batch_size=args.test_batch_size, shuffle=False)
    
    model = DGCNN_CAMGRAD(args).to(torch.device("cpu"))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, torch.device('cpu')))
    model = model.eval()

    point_sample, truth = next(iter(test_loader))
    point_sample = point_sample.permute(0, 2, 1)
    point_sample.requires_grad_()

    logits = model(point_sample)
    pred = logits.max(dim=1)
    loss = cal_loss(logits, truth)
    loss.backward()

    grad = point_sample.grad
    
    


if __name__ == "__main__":
    main()

    
