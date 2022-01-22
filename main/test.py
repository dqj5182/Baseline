import os
import argparse
import numpy as np
import torch
import shutil
import __init_path

from core.config import update_config, cfg
from core.logger import logger


parser = argparse.ArgumentParser(description='Test AnimateHuman')

parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--gpu', type=str, default='0,', help='assign multi-gpus by comma concat')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
    
np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")



from core.base import Tester


tester = Tester(args, load_dir=cfg.MODEL.weight_path)
print("===> Start testing...")
tester.test(0)

