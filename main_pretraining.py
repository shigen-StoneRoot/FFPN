import torch
import argparse
from monai import transforms
from monai.inferers import sliding_window_inference
from monai.metrics import RMSEMetric, MAEMetric, DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.transforms import RandRotate
import numpy as np
import random
from torch import optim
from functools import partial
from models import PreVesselFormer
from utils.data_utils import save_checkpoint, load_pretraining_train_val_loader
from utils.schedule_utils import adjust_learning_rate
from skimage import morphology
import os
from torch.nn import DataParallel
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import cycle
import tqdm
import time
from torch.cuda.amp import autocast, GradScaler
import math
from torch.nn import functional as F


rearrange_d = Rearrange('b c h w d -> b c d w h')



parser = argparse.ArgumentParser()
parser.add_argument('--ckdir', type=str, default="./pretraining_checkpoints")
parser.add_argument('--data_root_dir', type=str, default=r'./datasets/pretraining_datasets')
parser.add_argument('--dataset', type=str, default='IXI')


parser.add_argument('--image_size', type=tuple, default=(160, 160, 64))
parser.add_argument('--patch_size', type=tuple, default=(16, 16, 16))       # 7*7*4=196
parser.add_argument('--out_channels', type=int, default=2)

parser.add_argument('--feature_size', type=int, default=24)
parser.add_argument('--norm_name', type=str, default="instance")
parser.add_argument('--conv_block', type=bool, default=True)
parser.add_argument('--res_block', type=bool, default=True)
parser.add_argument('--dropout_rate', type=float, default=0.0)


parser.add_argument("--lr", default=5e-3, type=float, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=3e-5,
                    help='weight decay (default: 0.005)')
parser.add_argument('--warmup_epochs', type=int, default=20000)
parser.add_argument('--total_epochs', type=int, default=200000)
parser.add_argument('--reuse_ck', type=int, default=0)
parser.add_argument('--amp', type=bool, default=True)

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=1)

parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandRotate_prob", default=0.2, type=float, help="RandRotate aug probability")
parser.add_argument("--RandRotate_angle", default=[-0.25, 0.25], type=list, help="RandRotate angle range")
parser.add_argument("--GaussianNoise_prob", default=0.15, type=float, help="Gaussian Noise aug probability")
parser.add_argument("--GaussianSmooth_prob", default=0.2, type=float, help="Gaussian Smooth aug probability")
parser.add_argument("--GaussianSmooth_kernel", default=(0.5, 1.5), type=tuple, help="Gaussian Smooth kernel range")
parser.add_argument("--AdjustContrast_prob", default=0.15, type=float, help="Contrast aug probability")
parser.add_argument("--AdjustContrast_gamma", default=(0.65, 1.5), type=tuple, help="Adjust Contrast gamma range")

parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")

args = parser.parse_args()


def init_dataloader(args):
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "hessian", "label"]),
        transforms.AddChanneld(keys=["image", "hessian", "label"]),
        transforms.NormalizeIntensityd(keys=["image"], nonzero=True),
        transforms.RandWeightedCropd(keys=["image", "hessian", "label"], w_key="hessian", spatial_size=args.image_size),
        transforms.ResizeWithPadOrCropd(keys=["image", "hessian", "label"], spatial_size=args.image_size,
                                        mode=['constant', 'constant', 'constant']),
        transforms.RandFlipd(keys=["image", "hessian", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "hessian", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "hessian", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        transforms.RandRotated(keys=["image", "hessian", "label"], range_z=args.RandRotate_angle,
                               prob=args.RandRotate_prob, mode=["bilinear", "bilinear", "nearest"]),
                               
        transforms.RandGaussianNoised(keys="image", prob=args.GaussianNoise_prob),
        transforms.RandGaussianSmoothd(keys="image", prob=args.GaussianSmooth_prob,
                                       sigma_x=args.GaussianSmooth_kernel,
                                       sigma_y=args.GaussianSmooth_kernel,
                                       sigma_z=args.GaussianSmooth_kernel),
        transforms.RandAdjustContrastd(keys="image", prob=args.AdjustContrast_prob, gamma=args.AdjustContrast_gamma),


        transforms.RandScaleIntensityd(keys="image", factors=0.3, prob=args.RandScaleIntensityd_prob),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        transforms.ToTensord(keys=["image", "hessian", "label"])
    ])

    val_transform = train_transform

    train_loader, test_loader = load_pretraining_train_val_loader(
        ['OASIS', 'IXI', 'OpenNeuro', 'ADAM', 'TubeTK'], train_transform, val_transform, batch_size=args.batch_size,
        test_batch_size=args.test_batch_size, train_size=0.9, persistent_workers=False)
    return train_loader, test_loader


def init_model_optimizer(args):
    model = PreVesselFormer()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    scheduler = None

    return model, optimizer, scheduler


def run(args):
    torch.backends.cudnn.benchmark = True
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0')

    model, optimizer, scheduler = init_model_optimizer(args)
    train_loader, val_loader = init_dataloader(args)

    scaler = GradScaler()

    ds_weight = torch.tensor([1, 1 / 2, 1 / 4, 1 / 8, 1 / 16])
    ds_weight = ds_weight / ds_weight.sum()
    ds_weight = ds_weight.to(device)

    it = 0
    end_flag = False

    for epoch in range(10000000000):
        model.train()
        train_loss = 0.0

        iter_step = 0

        for batch_data in train_loader:
            adjust_learning_rate(optimizer, it, args)
            optimizer.zero_grad()
            img, hessian, label = rearrange_d(batch_data['image']).to(device), \
                                  rearrange_d(batch_data['hessian']).to(device), \
                                  rearrange_d(batch_data['label']).to(device)

            features, logits_rgn = model(img)

            l_seg = [loss_func(features[i], label) * ds_weight[i] for i in range(5)]
            l_seg = sum(l_seg)

            l_rgn = F.l1_loss(logits_rgn, regression_tar)

            ori_imgs_mip = torch.max(imgs * features[0], dim=4, keepdim=True)[0]
            logits_rgn_mip = torch.max(logits_rgn * features[0], dim=4, keepdim=True)[0]
            l_mip_consistency = F.l1_loss(logits_rgn_mip, ori_imgs_mip, reduction='none')
            l_mip_consistency = torch.mean(l_mip_consistency)

            loss = 0.4 * l_seg + 0.4 * l_rgn + 0.2 * l_mip_consistency

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            iter_step += 1
            it += 1

            del loss
            torch.cuda.empty_cache()

            if it >= args.total_epochs:
                end_flag = True
                break

        train_loss /= (iter_step + 0)
        print('epoch: ', epoch, 'iter: ', it, 'train_loss: ', train_loss)

        if it % 1000 == 0:
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    img, hessian, label = batch_data['image'].to(device), \
                        batch_data['hessian'].to(device), \
                        batch_data['label'].to(device)

                    features, logits_rgn = model(img)
                    l_seg = [loss_func(features[i], label) * ds_weight[i] for i in range(5)]
                    l_seg = sum(l_seg)
                    l_rgn = F.l1_loss(logits_rgn, regression_tar)
                    ori_imgs_mip = torch.max(imgs * features[0], dim=4, keepdim=True)[0]
                    logits_rgn_mip = torch.max(logits_rgn * features[0], dim=4, keepdim=True)[0]
                    l_mip_consistency = F.l1_loss(logits_rgn_mip, ori_imgs_mip, reduction='none')
                    l_mip_consistency = torch.mean(l_mip_consistency)

                    loss = 0.4 * l_seg + 0.4 * l_rgn + 0.2 * l_mip_consistency
                    print('epoch: ', epoch, 'iter: ', it, 'val_loss: ', loss)

            save_checkpoint(model, it, args, filename="model.pt")
        if end_flag:
            break


run(args)
