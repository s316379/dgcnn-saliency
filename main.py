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



criterion = cal_loss

DUMP_DIR = 'dump'
all_counters = np.zeros((40, 3), dtype=int)


def drop_points(pointclouds_pl, labels_pl, args, model, device):
    pointclouds_pl_adv = pointclouds_pl.clone().detach()
    pointclouds_pl_adv_np = pointclouds_pl_adv.clone().detach().cpu().numpy()
    # pointclouds_pl_adv_np.astype()

    for i in range(args.num_steps):
        pointclouds_pl_adv = torch.from_numpy(pointclouds_pl_adv_np).to(dtype=torch.float32).to(device=device)
        pointclouds_pl_adv.requires_grad_()

        logits = model(pointclouds_pl_adv)

        loss = cal_loss(logits, labels_pl)
        loss.backward()

        grad = pointclouds_pl_adv.grad
        grad_np = grad.clone().detach().numpy()
        ## median value
        sphere_core = np.median(pointclouds_pl_adv_np, axis=2, keepdims=True)
        
        sphere_r = np.sqrt(np.sum(np.square(pointclouds_pl_adv_np - sphere_core), axis=1)) ## BxN
        
        sphere_axis = pointclouds_pl_adv_np - sphere_core ## BxNx3

        if args.drop_neg:
            sphere_map = np.multiply(np.sum(np.multiply(grad_np, sphere_axis), axis=1), np.power(sphere_r, args.power))
        else:
            sphere_map = -np.multiply(np.sum(np.multiply(grad_np, sphere_axis), axis=1), np.power(sphere_r, args.power))

        drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1]-args.num_drops, axis=1)[:, -args.num_drops:]

        tmp = np.zeros((pointclouds_pl_adv_np.shape[0], 3, pointclouds_pl_adv_np.shape[2]-args.num_drops))
        for j in range(pointclouds_pl.shape[0]):
            tmp[j] = np.delete(pointclouds_pl_adv_np[j], drop_indice[j], axis=1) # along N points to delete
            
        pointclouds_pl_adv_np = tmp.copy()
        
    return pointclouds_pl_adv

def plot_natural_and_advsarial_samples_all_situation(pointclouds_pl, pointclouds_pl_adv, labels_pl, pred_val, pred_val_adv, all_counters, SHAPE_NAMES):
    for i in range(labels_pl.shape[0]):
        if labels_pl[i] == pred_val[i]:
            if labels_pl[i] != pred_val_adv[i]:
                img_filename = 'label_%s_advpred_%s_%d' % (SHAPE_NAMES[labels_pl[i]],
                                                        SHAPE_NAMES[pred_val_adv[i]], 
                                                        all_counters[labels_pl[i]][0])
                all_counters[labels_pl[i]][0] += 1
                img_filename = os.path.join(DUMP_DIR+'/pred_correct_adv_wrong', img_filename)
                pc_util.pyplot_draw_point_cloud_nat_and_adv(pointclouds_pl[i], pointclouds_pl_adv[i], img_filename)    
        else:
            if labels_pl[i] == pred_val_adv[i]:
                img_filename = 'label_%s_pred_%s_%d' % (SHAPE_NAMES[labels_pl[i]],
                                                        SHAPE_NAMES[pred_val[i]], 
                                                        all_counters[labels_pl[i]][1])
                all_counters[labels_pl[i]][1] += 1        
                img_filename = os.path.join(DUMP_DIR+'/pred_wrong_adv_correct', img_filename)
                
            else:
        
                img_filename = 'label_%s_pred_%s_advpred_%s_%d' % (SHAPE_NAMES[labels_pl[i]],
                                                        SHAPE_NAMES[pred_val[i]],
                                                        SHAPE_NAMES[pred_val_adv[i]],
                                                        all_counters[labels_pl[i]][2])
                all_counters[labels_pl[i]][2] += 1
                img_filename = os.path.join(DUMP_DIR+'/pred_wrong_adv_wrong', img_filename)
            
            pc_util.pyplot_draw_point_cloud_nat_and_adv(pointclouds_pl[i], pointclouds_pl_adv[i], img_filename)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40'])
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--num_drops', type=int, default=10, help='num of points to drop each step')
    parser.add_argument('--num_steps', type=int, default=10, help='num of steps to drop each step')
    parser.add_argument('--drop_neg', action='store_true',help='drop negative points')
    parser.add_argument('--power', type=int, default=6, help='x: -dL/dr*r^x')
    parser.add_argument('--subset', type=bool, default=False, help='If dataset has to be subesetted or not')


    args = parser.parse_args()

    ## 3 folders to store all the situations
    if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
    if not os.path.exists(DUMP_DIR+'/pred_correct_adv_wrong'): os.mkdir(DUMP_DIR+'/pred_correct_adv_wrong')
    if not os.path.exists(DUMP_DIR+'/pred_wrong_adv_correct'): os.mkdir(DUMP_DIR+'/pred_wrong_adv_correct')
    if not os.path.exists(DUMP_DIR+'/pred_wrong_adv_wrong'): os.mkdir(DUMP_DIR+'/pred_wrong_adv_wrong')

    # download modelnet if not exists
    download()

    SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'dgcnn-master/pytorch/data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

    modelnet = ModelNet40(partition='test', num_points=1024)
    indices = np.random.choice(len(modelnet), 24, replace=False)
    subset_modelnet = Subset(modelnet, indices)
    
    dataset = modelnet if args.subset == True else subset_modelnet

    test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    model = DGCNN_CAMGRAD(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, device))
    model = model.eval()

    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []

    test_acc_adv = 0.0
    count_adv = 0.0
    test_true_adv = []
    test_pred_adv = []
    
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        data.requires_grad_()
        batch_size = data.size()[0]

        adversial_data = drop_points(data, label, args, model, device)

        # NATURAL DATA
        logits = model(data).to(device)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

        # ADVERSIAL DATA
        logits_adv = model(adversial_data)        
        preds_adv = logits_adv.max(dim=1)[1]
        test_true_adv.append(label.cpu().numpy())
        test_pred_adv.append(preds_adv.detach().cpu().numpy())

        plot_natural_and_advsarial_samples_all_situation(data.detach().numpy(), adversial_data.detach().numpy(), label.detach().numpy(), preds.detach().numpy(), preds_adv.detach().numpy(), all_counters, SHAPE_NAMES)

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test natural:: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    cprint.info(outstr)

    test_true_adv = np.concatenate(test_true_adv)
    test_pred_adv = np.concatenate(test_pred_adv)
    test_acc_adv = metrics.accuracy_score(test_true_adv, test_pred_adv)
    avg_per_class_acc_adv = metrics.balanced_accuracy_score(test_true_adv, test_pred_adv)
    outstr_adv = 'Test adversial:: test acc: %.6f, test avg acc: %.6f'%(test_acc_adv, avg_per_class_acc_adv)
    cprint.info(outstr_adv)
    
    


if __name__ == "__main__":
    main()

    
