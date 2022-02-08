import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
import cv2

from torch.utils.data import Dataset

from core.config import cfg
from core.logger import logger
from img_utils import load_img, annToMask
from coord_utils import generate_joint_heatmap, sampling_non_joint, image_bound_check
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smpl_param_processing, flip_joint, transform_joint_to_other_db
from human_models import smpl

from torchvision.transforms import Normalize

from vis_utils import vis_keypoints, vis_keypoints_with_skeleton, vis_3d_pose, vis_heatmaps, save_obj

from _img_utils import get_single_image_crop_demo

class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_joint_cam = False
        self.has_smpl_param = False
        self.normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = copy.deepcopy(self.datalist[index])
        
        img_path = data['img_path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bbox = data['bbox']
        
        bbox[0] = bbox[0] + bbox[2]/2; bbox[1] = bbox[1] + bbox[3]/2

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=None,
            scale=1.0,
            crop_size=cfg.MODEL.input_img_shape[0])

        if self.has_joint_cam:
            joint_cam = data['joint_cam']
            joint_cam = joint_cam - joint_cam[self.root_joint_idx]
        else:
            joint_cam = np.zeros((smpl.joint_num, 3))

        batch = {
            'img': norm_img,
            'joint_cam': joint_cam * 1000
        }
        
        return batch
    
    def get_smpl_coord(self, smpl_pose, smpl_shape):
        root_pose, body_pose, smpl_shape = torch.tensor(smpl_pose[:3]).reshape(1,-1), torch.tensor(smpl_pose[3:]).reshape(1,-1), torch.tensor(smpl_shape).reshape(1,-1)
        output = smpl.layer['neutral'](betas=smpl_shape, body_pose=body_pose, global_orient=root_pose)
        smpl_mesh_cam = output.vertices[0].numpy()
        smpl_joint_cam = np.dot(smpl.joint_regressor, smpl_mesh_cam)
        smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[smpl.root_joint_idx]
        return smpl_mesh_cam, smpl_joint_cam