import os
import sys
import time
import math
import numpy as np
import cv2
import shutil
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchgeometry as tgm

from core.config import cfg


def sample_image_feature(img_feat, xy, width, height):
    x = xy[:,0] / width * 2 - 1
    y = xy[:,1] / height * 2 - 1
    grid = torch.stack((x,y),1)[None,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[0,:,:,0] # (channel_dim, sampling points)
    img_feat = img_feat.permute(1,0)
    return img_feat


def lr_check(optimizer, epoch):
    base_epoch = 5
    if False and epoch <= base_epoch:
        lr_warmup(optimizer, cfg.TRAIN.lr, epoch, base_epoch)

    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
    print(f"Current epoch {epoch}, lr: {curr_lr}")


def lr_warmup(optimizer, lr, epoch, base):
    lr = lr * (epoch / base)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.acc += time.time() - self.t0  # cacluate time diff

    def reset(self):
        self.acc = 0

    def print(self):
        return round(self.acc, 2)


def stop():
    sys.exit()


def check_data_parallel(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model):
    total_params = []
    for module in model.trainable_modules:
        total_params += list(module.parameters())

    optimizer = None
    if cfg.TRAIN.optimizer == 'sgd':
        optimizer = optim.SGD(
            total_params,
            lr=cfg.TRAIN.lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay,
            nesterov=cfg.TRAIN.nesterov
        )
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            total_params,
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adam':
        optimizer = optim.Adam(
            total_params,
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adamw':
        optimizer = optim.AdamW(
            total_params,
            lr=cfg.TRAIN.lr,
            weight_decay=0.1
        )

    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)

    return scheduler


def save_checkpoint(states, epoch, is_best=None):
    file_name = f'epoch_{epoch}.pth.tar'
    output_dir = cfg.checkpoint_dir
    if states['epoch'] == cfg.TRAIN.end_epoch:
        file_name = 'final.pth.tar'
    torch.save(states, os.path.join(output_dir, file_name))

    if is_best:
        torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)

def calculate_iou_matrix(boxs1, boxs2):
    boxs1, boxs2 = torch.tensor(boxs1).clone(), torch.tensor(boxs2).clone()

    boxs1[:,2] = boxs1[:,0] + boxs1[:,2]
    boxs1[:,3] = boxs1[:,1] + boxs1[:,3]
    boxs2[:,2] = boxs2[:,0] + boxs2[:,2]
    boxs2[:,3] = boxs2[:,1] + boxs2[:,3]

    N, _ = boxs1.shape      # [N, 4]
    M, _ = boxs2.shape      # [M, 4]
    
    boxs1 = boxs1.view(N, 1, 4).expand(N, M, 4)     # [N, M, 4]
    boxs2 = boxs2.view(1, M, 4).expand(N, M, 4)     # [N, M, 4]

    iw_max = torch.min(boxs1[:, :, 2], boxs2[:, :, 2])
    iw_min = torch.max(boxs1[:, :, 0], boxs2[:, :, 0])
    iw = (iw_max - iw_min)
    iw[iw < 0] = 0

    ih_max = torch.min(boxs1[:, :, 3], boxs2[:, :, 3])
    ih_min = torch.max(boxs1[:, :, 1], boxs2[:, :, 1])
    ih = (ih_max - ih_min)
    ih[ih < 0] = 0
    
    boxs1_area = ((boxs1[:, :, 2] - boxs1[:, :, 0]) * (boxs1[:, :, 3] - boxs1[:, :, 1]))
    boxs2_area = ((boxs2[:, :, 2] - boxs2[:, :, 0]) * (boxs2[:, :, 3] - boxs2[:, :, 1]))

    inter = iw * ih                             # [N, M]
    union = boxs1_area + boxs2_area - inter     # [N, M]
    iou_matrix = inter / union                  # [N, M]

    return iou_matrix

def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
    
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

