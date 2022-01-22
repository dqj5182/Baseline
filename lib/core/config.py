import os
import os.path as osp
import shutil

import yaml
from easydict import EasyDict as edict
import datetime


def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)


cfg = edict()

""" Directory """
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)

cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
cfg.res_dir = osp.join(cfg.output_dir, 'result')
cfg.log_dir = osp.join(cfg.output_dir, 'log')
cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')

print("Experiment Data on {}".format(cfg.output_dir))
init_dirs([cfg.output_dir, cfg.log_dir, cfg.graph_dir, cfg.vis_dir, cfg.checkpoint_dir])

""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_list = []
cfg.DATASET.train_partition = [1.0]
cfg.DATASET.test_list = []
cfg.DATASET.make_same_len = False
cfg.DATASET.workers = 16
cfg.DATASET.random_seed = 123

""" Model """
cfg.MODEL = edict()
cfg.MODEL.backbone = 'resnet50'
cfg.MODEL.use_upsampling_layer = False
cfg.MODEL.input_img_shape = (256, 192)
cfg.MODEL.img_feat_shape = (64, 48)
cfg.MODEL.projector_hidden_dim = 256
cfg.MODEL.projector_out_dim = 256
cfg.MODEL.predictor_hidden_dim = 256
cfg.MODEL.predictor_pose_feat_dim = 128
cfg.MODEL.predictor_shape_feat_dim = 64
cfg.MODEL.weight_path = ''

""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.freeze_backbone = False
cfg.TRAIN.batch_size = 8
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 30
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 1e-3
cfg.TRAIN.lr_step = [20]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'rmsprop'
cfg.TRAIN.print_freq = 10
cfg.TRAIN.non_joints_num = 24
cfg.TRAIN.heatmap_sigma = 10
cfg.TRAIN.use_pseudo_GT = True
cfg.TRAIN.vis = False

cfg.TRAIN.inter_joint_loss_weight = 1.0
cfg.TRAIN.intra_joint_loss_weight = 1.0
cfg.TRAIN.joint_loss_weight = 1.0
cfg.TRAIN.proj_loss_weight = 1.0
cfg.TRAIN.pose_loss_weight = 1.0
cfg.TRAIN.shape_loss_weight = 1.0
cfg.TRAIN.prior_loss_weight = 1.0e-4

""" Augmentation """
cfg.AUG = edict()
cfg.AUG.scale_factor = 0
cfg.AUG.rot_factor = 0
cfg.AUG.color_factor = 0
cfg.AUG.flip = False

""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 16
cfg.TEST.shuffle = False
cfg.TEST.vis = False
cfg.TEST.vis_freq = 50

""" CAMERA """
cfg.CAMERA = edict()
cfg.CAMERA.focal = (5000, 5000)
cfg.CAMERA.princpt = (cfg.MODEL.input_img_shape[1]/2, cfg.MODEL.input_img_shape[0]/2)
cfg.CAMERA.camera_3d_size = 2.5


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            cfg[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, yaml.SafeLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


