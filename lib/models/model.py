import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy

from models import PoseResNet, PoseHighResolutionNet, Predictor
from core.config import cfg
from core.logger import logger
from collections import OrderedDict
from funcs_utils import sample_image_feature, rot6d_to_axis_angle
from human_models import smpl

class Model(nn.Module):
    def __init__(self, backbone, head):
        super(Model, self).__init__()
        self.backbone = backbone
        self.head = head
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
        
        if cfg.TRAIN.freeze_backbone:
            self.trainable_modules = [self.head]
        else:
            self.trainable_modules = [self.backbone, self.head]


    def forward(self, inp_img):
        batch_size = inp_img.shape[0]
        img_feat = self.backbone(inp_img)

        smpl_pose, smpl_shape, cam_trans = self.head(img_feat)

        smpl_pose = rot6d_to_axis_angle(smpl_pose.reshape(-1,6)).reshape(batch_size,-1)
        cam_trans = self.get_camera_trans(cam_trans)
        joint_proj, joint_cam, mesh_cam = self.get_coord(smpl_pose[:,:3], smpl_pose[:,3:], smpl_shape, cam_trans)
        
        return mesh_cam, joint_cam, joint_proj, smpl_pose, smpl_shape


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.CAMERA.focal[0]*cfg.CAMERA.focal[1]*cfg.CAMERA.camera_3d_size*cfg.CAMERA.camera_3d_size/(cfg.MODEL.input_img_shape[0]*cfg.MODEL.input_img_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans
    
    
    def get_coord(self, smpl_root_pose, smpl_pose, smpl_shape, cam_trans):
        batch_size = smpl_root_pose.shape[0]
        
        output = self.smpl_layer(global_orient=smpl_root_pose, body_pose=smpl_pose, betas=smpl_shape)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(smpl.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_joint_idx = smpl.root_joint_idx
        
        # project 3D coordinates to 2D space
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[0] + cfg.CAMERA.princpt[0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[1] + cfg.CAMERA.princpt[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam
    
    
def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass


def get_model(is_train):
    if cfg.MODEL.backbone == 'resnet50':
        backbone = PoseResNet(50, do_upsampling=cfg.MODEL.use_upsampling_layer)
        pretrained = 'data/base_data/backbone_models/cls_resnet_50_imagenet.pth'
        if cfg.MODEL.use_upsampling_layer: 
            cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//4, cfg.MODEL.input_img_shape[1]//4)
            backbone_out_dim = 256
        else: 
            cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//32, cfg.MODEL.input_img_shape[1]//32)
            backbone_out_dim = 2048
    elif cfg.MODEL.backbone == 'hrnetw32':
        backbone = PoseHighResolutionNet(do_upsampling=cfg.MODEL.use_upsampling_layer)
        cfg.MODEL.img_feat_shape = (cfg.MODEL.input_img_shape[0]//4, cfg.MODEL.input_img_shape[1]//4)
        pretrained = 'data/base_data/backbone_models/cls_hrnet_w32_imagenet.pth'
        if cfg.MODEL.use_upsampling_layer: backbone_out_dim = 480
        else: backbone_out_dim = 32
    
    head = Predictor(backbone_out_dim,cfg.MODEL.predictor_hidden_dim)
    
    if is_train:
        backbone.init_weights(pretrained)    
        head.apply(init_weights)
            
    model = Model(backbone, head)
    return model


def transfer_backbone(model, checkpoint):    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if 'backbone' in k:
            name = k.replace('backbone.', '')
            new_state_dict[name] = v
            
    model.backbone.load_state_dict(new_state_dict)