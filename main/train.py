import os
import argparse
import torch
import numpy as np
import __init_path
import shutil

from funcs_utils import save_checkpoint, check_data_parallel
from core.config import cfg, update_config
from core.logger import logger

parser = argparse.ArgumentParser(description='Train AnimateHuman')

parser.add_argument('--resume_training', action='store_true', help='resume training')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')

args = parser.parse_args()
if args.cfg: update_config(args.cfg)

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")

from core.base import Trainer, Tester

trainer = Trainer(args, load_dir=cfg.MODEL.weight_path)
tester = Tester(args)

logger.info(f"===> Start training...")
for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    trainer.train(epoch)
    trainer.lr_scheduler.step()
    
    if len(cfg.DATASET.test_list) > 0:
        tester.test(epoch, current_model=trainer.model)    
    
    is_best = None
    if len(trainer.error_history['pa_mpjpe']) > 0:
        is_best = tester.pa_mpjpe < min(trainer.error_history['pa_mpjpe'])

    tester.save_history(trainer.loss_history, trainer.error_history, epoch) 
        
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': check_data_parallel(trainer.model.state_dict()),
        'optim_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'train_log': trainer.loss_history,
        'test_log': trainer.error_history
    }, epoch, is_best)
    
    
    



