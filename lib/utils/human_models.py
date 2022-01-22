import numpy as np
import torch
import os.path as osp
import json
import smplx

class SMPL(object):
    def __init__(self):
        self.model_path = osp.join('data', 'base_data', 'human_models')
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(self.model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(self.model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(self.model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)
        self.shape_param_dim = 10

        self.joint_num = 24 
        self.joints_name = (
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 
        'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.part_segments_color = ('silver', 'blue', 'green', 'salmon', 'turquoise', 'olive', 'lavender', 'darkblue', 'lime', 'khaki', 'cyan', 'darkgreen',
                                    'beige', 'coral', 'crimson', 'red', 'aqua', 'chartreuse', 'indigo', 'teal', 'violet', 'orchid', 'orange', 'gold')
        self.flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.skeleton = (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
        (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.root_joint_idx = self.joints_name.index('Pelvis')
        
        
        self.h36m_joint_regressor = np.load(osp.join('data', 'base_data', 'J_regressor_h36m_smpl.npy'))
        self.h36m_root_joint_idx = 0
        self.h36m_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.h36m_eval_joints = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        
class MANO(object):
    def __init__(self):
        self.model_path = osp.join('data', 'base_data', 'human_models')
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        self.layer = {'right': smplx.create(self.model_path, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False, **self.layer_arg), 'left': smplx.create(cfg.human_model_path, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False, **self.layer_arg)}
        
        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10

        if torch.sum(torch.abs(self.layer['left'].shapedirs[:,0,:] - self.layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:,0,:] *= -1

        self.joint_num = 16
        self.joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.root_joint_idx = self.orig_joints_name.index('Wrist')
        self.flip_pairs = ()
        self.joint_regressor = self.layer['right'].J_regressor.numpy()

smpl = SMPL()
#mano = MANO()