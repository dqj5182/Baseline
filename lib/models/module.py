import torch
import torch.nn as nn
from torch.nn import functional as F

from models import make_linear_layers, make_conv_layers, make_deconv_layers
from human_models import smpl

class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = make_linear_layers([in_dim, hidden_dim], relu_final=False, use_bn=False)
        self.drop1 = nn.Dropout()
        self.fc2 = make_linear_layers([hidden_dim, hidden_dim], relu_final=False, use_bn=False)
        self.drop2 = nn.Dropout()
        self.pose_out = make_linear_layers([hidden_dim,smpl.joint_num*6], relu_final=False, use_bn=False)
        self.shape_out = make_linear_layers([hidden_dim,smpl.shape_param_dim], relu_final=False, use_bn=False)
        self.cam_out = make_linear_layers([hidden_dim,3], relu_final=False, use_bn=False)

    def forward(self, x):  
        x = x.mean((2,3))

        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        pose = self.pose_out(x)
        shape = self.shape_out(x)
        cam_trans = self.cam_out(x)

        return pose, shape, cam_trans