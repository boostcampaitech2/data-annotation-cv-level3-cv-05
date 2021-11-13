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
from dataset import SceneTextDataset
from model import EAST

from dataset import ValidSceneTextDataset, ValidEASTDataset, collate_fn
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

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_interval', type=int, default=10)
    parser.add_argument('--wandb_skip', type=bool, default=False)
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
                learning_rate, max_epoch, save_interval, wandb_interval, wandb_skip, seed, exp_name):
    if exp_name is None:
        raise BaseException("You must set 'exp_name'.")
    else:
        NAME = exp_name

    set_seed(seed)
    dataset = SceneTextDataset(data_dir, split='new_train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)

    train_dataset = dataset
    valid_dataset = ValidSceneTextDataset(data_dir, split='new_valid', image_size=image_size, crop_size=input_size)
    valid_dataset = ValidEASTDataset(valid_dataset)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(valid_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # Set wandb
    if wandb_skip is False:
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

    model.train()
    for epoch in range(max_epoch):
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

                if wandb_skip is False and (step + 1) % wandb_interval == 0:
                    wandb.log({ "train/loss": loss.item(), 
                                "train/cls_loss": train_dict['Cls loss'],
                                "train/angle_loss": train_dict['Angle loss'],
                                "train/iou_loss": train_dict['IoU loss'],
                                "epoch":epoch+1}, step=epoch*num_batches+step)

        scheduler.step()

        val_epoch_loss = 0
        val_cls_loss = 0
        val_angle_loss = 0
        val_iou_loss = 0
        pred_bboxes_dict = dict()
        gt_bboxes_dict = dict()
        transcriptions_dict = dict()
        with tqdm(total=val_num_batches) as pbar:
            model.eval()
            for img, gt_score_map, gt_geo_map, roi_mask, transcriptions, word_bboxes, image_fnames in valid_loader:
                pbar.set_description('[Valid {}]'.format(epoch + 1))

                img, gt_score_map, gt_geo_map, roi_mask = (img.to(device), gt_score_map.to(device),
                                               gt_geo_map.to(device), roi_mask.to(device))
                pred_score_map, pred_geo_map = model(img)

                loss, values_dict = model.criterion(gt_score_map, pred_score_map, gt_geo_map, pred_geo_map,
                                           roi_mask)

                extra_info = dict(**values_dict, score_map=pred_score_map, geo_map=pred_geo_map)

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                loss_val = loss.item()
                val_epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

                val_cls_loss += val_dict['Cls loss']
                val_angle_loss += val_dict['Angle loss']
                val_iou_loss += val_dict['IoU loss']

                if epoch > -1:
                    orig_imgs, orig_size = [], []
                    input_size = 256
                    for image_fname in image_fnames:
                        image_fpath = osp.join(data_dir, f'images/{image_fname}')
                        image_ = cv2.imread(image_fpath)[:, :, ::-1]
                        orig_imgs.append(image_)
                        orig_size.append(max(image_.shape[:2]))
                    pred_bboxes = detect(model, orig_imgs, input_size)

                    pred_bboxes *= np.array(orig_size) / input_size
                    for p, g in zip(pred_bboxes, word_bboxes):
                        print(p)#, g)
                        print()

                    for image_fname, pred_bbox, gt_bbox, transcription in zip(image_fnames, pred_bboxes, word_bboxes, transcriptions):
                        pred_bboxes_dict[image_fname] = pred_bbox
                        gt_bboxes_dict[image_fname] = gt_bbox
                        transcriptions_dict[image_fname] = transcription
            if epoch > -1:
                result_dict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)
                print(result_dict['total'])
                if wandb_skip is False:
                    wandb.log({ "val/precision": result_dict['total']['precision'],
                            "val/recall": result_dict['total']['recall'],
                            "val/hmean": result_dict['total']['hmean'],
                            "epoch":epoch+1})
        if wandb_skip is False:            
            wandb.log({ "val/loss": val_epoch_loss / val_num_batches,
                        "val/cls_loss": val_cls_loss / val_num_batches,
                        "val/angle_loss": val_angle_loss / val_num_batches,
                        "val/iou_loss": val_iou_loss / val_num_batches,
                        "epoch":epoch+1})

        print('Train mean loss: {:.4f} | Val mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, val_epoch_loss / val_num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        
        print()


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
