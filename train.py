import os
import os.path as osp
import time
import math
import numpy as np
import random
from datetime import timedelta
from argparse import ArgumentParser
import wandb

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, ValidSceneTextDataset
from model import EAST
from detect import get_bboxes
from deteval import calc_deteval_metrics

from dataset import ValidSceneTextDataset
import cv2
from detect import detect
from deteval import calc_deteval_metrics

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=7)

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--exp_name', type=str)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed : {seed}")


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_interval, seed, exp_name):
    if exp_name is None:
        raise BaseException("You must set 'exp_name'.")
    else:
        NAME = exp_name

    set_seed(seed)
    train_dataset = SceneTextDataset(data_dir, split='new_train', image_size=image_size, crop_size=input_size)
    train_dataset.load_image()
    train_dataset = EASTDataset(train_dataset)
    print(f"Load train data {len(train_dataset)}")
    valid_dataset = ValidSceneTextDataset(data_dir, split='new_valid', image_size=image_size)
    valid_dataset.load_image()
    print(f"Load valid data {len(valid_dataset)}")

    num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(valid_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=ValidSceneTextDataset.collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Set wandb
    config = {
        'image_size':image_size, 'input_size':input_size, 'num_workers':num_workers, 'batch_size':batch_size,
        'learning_rate':learning_rate, 'epochs':max_epoch, 'seed':seed
    }
    wandb.init(project='OCR', entity='friends', config=config, name = NAME)
    wandb.define_metric("epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("val/loss", summary="min")
    wandb.watch(model)

    metric = 0
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for step, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict)

                if (step + 1) % wandb_interval == 0:
                    wandb.log({ "train/loss": loss.item(), 
                                "train/cls_loss": train_dict['Cls loss'],
                                "train/angle_loss": train_dict['Angle loss'],
                                "train/iou_loss": train_dict['IoU loss'],
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "epoch":epoch+1}, step=epoch*num_batches+step)

        scheduler.step()

        if epoch < 0:
            continue
        val_epoch_loss = 0
        val_cls_loss = 0
        val_angle_loss = 0
        val_iou_loss = 0            
        pred_bboxes_dict = dict()
        gt_bboxes_dict = dict()
        transcriptions_dict = dict()
        with tqdm(total=val_num_batches) as pbar:
            with torch.no_grad():
                model.eval()
                for step, (img, gt_score_map, gt_geo_map, roi_mask, vertices, orig_sizes, labels, transcriptions, fnames) in enumerate(valid_loader):
                    pbar.set_description('[Valid {}]'.format(epoch + 1))

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    
                    score_maps, geo_maps = extra_info['score_map'], extra_info['geo_map']
                    score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()
                    
                    by_sample_bboxes = []
                    for i, (score_map, geo_map, orig_size, vertice, transcription, fname) in enumerate(zip(score_maps, geo_maps, orig_sizes, vertices, transcriptions, fnames)):
                        map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.25 * image_size / max(orig_size))
                        if orig_size[0] > orig_size[1]:
                            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
                        else:
                            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

                        bboxes = get_bboxes(score_map, geo_map)
                        if bboxes is None:
                            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
                        else:
                            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
                        
                        pred_bboxes_dict[fname] = bboxes
                        gt_bboxes_dict[fname] = vertice
                        transcriptions_dict[fname] = transcription
                    
                    loss_val = loss.item()
                    if loss_val is not None:
                        val_epoch_loss += loss_val

                    pbar.update(1)
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'],
                        'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(val_dict)

                    if val_dict['Cls loss'] is not None:
                        val_cls_loss += val_dict['Cls loss']
                        val_angle_loss += val_dict['Angle loss']
                        val_iou_loss += val_dict['IoU loss']

        resDict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)   
        wandb.log({ "val/loss": val_epoch_loss / val_num_batches,
                    "val/cls_loss": val_cls_loss / val_num_batches,
                    "val/angle_loss": val_angle_loss / val_num_batches,
                    "val/iou_loss": val_iou_loss / val_num_batches,
                    "val/recall": resDict['total']['recall'],
                    "val/precision": resDict['total']['precision'],
                    "val/f1_score": resDict['total']['hmean'],
                    "epoch":epoch+1})
        print(resDict['total'])
        print('Train mean loss: {:.4f} | Val mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, val_epoch_loss / val_num_batches, timedelta(seconds=time.time() - epoch_start)))
        if metric < resDict['total']['hmean']:
            metric = resDict['total']['hmean']
            if not osp.exists('best_models'):
                os.makedirs('best_models')
            ckpt_fpath = osp.join('best_models', 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        print()


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
