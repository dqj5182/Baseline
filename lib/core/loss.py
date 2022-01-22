import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import cfg
from core.prior import MaxMixturePrior

class CoordLoss(nn.Module):
    def __init__(self, has_valid=False):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, target_valid=None):
        if self.has_valid:
            pred, target = pred * target_valid[...,None], target * target_valid[...,None]

        loss = self.criterion(pred, target)
        return loss
    
class ParamLoss(nn.Module):
    def __init__(self, has_valid=False):
        super(ParamLoss, self).__init__()
        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, param_out, param_gt, valid=None):
        valid = valid.view(-1)
        
        if self.has_valid:
            param_out, param_gt = param_out * valid[:,None], param_gt * valid[:,None]
        
        loss = self.criterion(param_out, param_gt)
        return loss

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, param):
        zeros = torch.zeros_like(param, device='cuda')
        loss = self.criterion(param, zeros)
        return loss.mean()

class HeatmapMSELoss(nn.Module):
    def __init__(self, has_valid):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.has_valid = has_valid

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        target_weight = target_weight.reshape((batch_size, num_joints, 1))
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.has_valid:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
      
class Joint2NonJointLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(Joint2NonJointLoss, self).__init__()
        self.temperature = temperature
        self.criterion = SupConLoss()

    def forward(self, output, target):
        batch_size, joint_num, n_view, feat_dim = output.shape
        
        output = output.reshape(batch_size*joint_num, n_view, feat_dim)
        target = target.reshape(batch_size*joint_num)    
        
        # remove not visible
        target_valid = (target != 0)
        output = output[target_valid]
        target = target[target_valid]
        labels = torch.zeros((len(target),), device='cuda')
        labels[target==1] = 1
        
        loss = self.criterion(output, labels)
        return loss
    
    
class Joint2JointLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(Joint2JointLoss, self).__init__()
        self.temperature = temperature
        self.criterion = SupConLoss()

    def forward(self, output, target):
        # remove non-joint
        output = output[:, :-cfg.TRAIN.non_joints_num, ...]
        target = target[:, :-cfg.TRAIN.non_joints_num]
        
        if (target<0).sum() > 0 : assert 0
        
        batch_size, joint_num, n_view, feat_dim = output.shape
        
        labels = torch.arange(joint_num, device='cuda')
        labels = torch.repeat_interleave(labels[None,:], batch_size, dim=0)
        
        output = output.reshape(batch_size*joint_num, n_view, feat_dim)
        labels = labels.reshape(batch_size*joint_num)    
        
        # remove not visible
        target_valid = (target.reshape(batch_size*joint_num) != 0)
        output = output[target_valid]
        labels = labels[target_valid]
        
        loss = self.criterion(output, labels)
        return loss
    
    
class JointContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(JointContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = SupConLoss()

    def forward(self, output, target):        
        batch_size, joint_num, n_view, feat_dim = output.shape
        
        labels = torch.arange(joint_num, device='cuda')
        labels = torch.repeat_interleave(labels[None,:], batch_size, dim=0)
        labels[:, -cfg.TRAIN.non_joints_num:, ...] = joint_num - cfg.TRAIN.non_joints_num # dummy label : 24
           
        output = output.reshape(batch_size*joint_num, n_view, feat_dim)
        target = target.reshape(batch_size*joint_num)   
        labels = labels.reshape(batch_size*joint_num)    
        
        # remove not visible
        target_valid = (target != 0)
        output = output[target_valid]
        labels = labels[target_valid]
        
        loss = self.criterion(output, labels)
        return loss
    
    
class PriorLoss(nn.Module):
    def __init__(self):
        super(PriorLoss, self).__init__()
        self.pose_prior = MaxMixturePrior(prior_folder='data/base_data/pose_prior', num_gaussians=8, dtype=torch.float32).cuda()
        self.pose_prior_weight = 4.78 ** 2
        self.shape_prior_weight = 5 ** 2
        self.angle_prior_weight = 15.2 **2
        
    def forward(self, body_pose, betas):
        # Pose prior loss
        pose_prior_loss = self.pose_prior_weight * self.pose_prior(body_pose, betas)

        # Angle prior for knees and elbows
        angle_prior_loss = self.angle_prior_weight * self.angle_prior(body_pose).sum(dim=-1)

        # Regularizer to prevent betas from taking large values
        shape_prior_loss = self.shape_prior_weight * (betas ** 2).sum(dim=-1)
        
        loss = pose_prior_loss + angle_prior_loss + shape_prior_loss
        return loss.mean()
    
    def angle_prior(self, pose):
        """
        Angle prior that penalizes unnatural bending of the knees and elbows
        """
        # We subtract 3 because pose does not include the global rotation of the model
        return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2
    
    
def get_loss():
    loss = {}
    loss['joint_cam'] = CoordLoss(has_valid=True)
    loss['smpl_joint_cam'] = CoordLoss(has_valid=True)
    loss['joint_proj'] = CoordLoss(has_valid=True)
    loss['pose_param'] = ParamLoss(has_valid=True)
    loss['shape_param'] = ParamLoss(has_valid=True)
    loss['prior'] = PriorLoss()
    return loss
