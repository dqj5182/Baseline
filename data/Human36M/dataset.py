import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
from pycocotools.coco import COCO # Python library for COCO format

from core.config import cfg
from coord_utils import world2cam, cam2pixel, process_bbox
from base_dataset import BaseDataset

class Human36M(BaseDataset):
    def __init__(self, transform, data_split):
        super(Human36M, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'Human36M', 'images') # outputs 'data/Human36M/images'
        self.annot_path = osp.join('data', 'Human36M', 'annotations') # outputs 'data/Human36M/annotations'

        self.joint_set = {
            'name': 'Human36M',
            'joint_num': 17,
            'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
            'flip_pairs': ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)),
            'skeleton': ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        } # Base Human36M dataset configuration
        
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis') # Set root joint as 'Pelvis' (also base Human36M configuration)
        
        self.has_joint_cam = True
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT # SMPL parameter
        self.datalist = self.load_data()
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8] # 5 subjects for training
        elif self.data_split == 'test':
            subject = [9,11] # 2 subjects for testing
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject() # list of subjects (total 7)
        sampling_ratio = self.get_subsampling_ratio()
        
        db = COCO()
        cameras = {} # camera dictionary
        joints = {} # joint dictionary
        smpl_params = {} # smpl parameters dictionary
        
        # For each subject
        for subject in subject_list:
            
            # load data on annotation
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f) # save the data into variable 'annot'
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
                    
            # load data on camera
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
                
            # load data on joint
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            if cfg.TRAIN.use_pseudo_GT:
                # load smpl parameter
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPL_NeuralAnnot.json'),'r') as f:
                    smpl_params[str(subject)] = json.load(f)
            else:
                smpl_params = None
        
        db.createIndex()
        
        
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx']
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
            if smpl_params is not None:
                try:
                    smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
                except KeyError:
                    smpl_param = None
            else:
                smpl_param = None
                
            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
            
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)[:,:2]
            joint_valid = np.ones((self.joint_set['joint_num'],))
            
            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img, 
                'joint_cam': joint_cam/1000,
                'joint_valid': joint_valid,
                'smpl_param': smpl_param,
                'cam_param': cam_param
                })

        return datalist
