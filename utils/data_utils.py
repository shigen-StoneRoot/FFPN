from monai import data
import random
import os
from monai.data.utils import pad_list_data_collate
import math
import warnings
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
import torch
from typing import Any, Callable, List, Sequence, Tuple, Union
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices


def load_pretraining_train_val_datasets(dataset_names, train_transform, val_transform,
                                        root_dir=r'./pretraining datasets', train_size=0.9):
    train_datasets_list = []
    val_datasets_list = []

    for dataset in dataset_names:
        cur_ds_img_dir = os.path.join(root_dir, dataset, 'image')
        cur_ds_gt_dir = os.path.join(root_dir, dataset, 'coarse_gt')
        cur_ds_hs_dir = os.path.join(root_dir, dataset, 'hessian')

        cur_ds_sub_list = os.listdir(cur_ds_img_dir)

        cur_ds_train_subs = cur_ds_sub_list[: int(train_size * len(cur_ds_sub_list))]
        cur_ds_val_subs = cur_ds_sub_list[int(train_size * len(cur_ds_sub_list)):]

        cur_ds_train_list = [{'image': os.path.join(cur_ds_img_dir, sub),
                              'hessian': os.path.join(cur_ds_gt_dir, sub),
                              'label': os.path.join(cur_ds_gt_dir, sub)} for sub in cur_ds_train_subs]

        cur_ds_val_list = [{'image': os.path.join(cur_ds_img_dir, sub),
                            'hessian': os.path.join(cur_ds_gt_dir, sub),
                            'label': os.path.join(cur_ds_gt_dir, sub)} for sub in cur_ds_val_subs]

        train_datasets_list.extend(cur_ds_train_list)
        val_datasets_list.extend(cur_ds_val_list)

    train_dataset = data.Dataset(data=train_datasets_list, transform=train_transform)
    test_dataset = data.Dataset(data=val_datasets_list, transform=val_transform)

    return train_dataset, test_dataset


def load_pretraining_train_val_loader(dataset_names, train_transform, val_transform,
                                      root_dir=r'./pretraining datasets',
                                      shuffle=True, train_size=0.9, batch_size=2, num_workers=4,
                                      pin_memory=True, persistent_workers=True, test_batch_size=1):

    train_dataset, val_dataset = load_pretraining_train_val_datasets(
        dataset_names=dataset_names, train_transform=train_transform,
        val_transform=val_transform, root_dir=root_dir, train_size=train_size)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=pad_list_data_collate
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader


def save_checkpoint(model, epoch, args, filename="model.pt"):
    state_dict = model.state_dict()
    save_dict = {"state_dict": state_dict}
    filename = os.path.join(args.ckdir, str(epoch) + '_' + filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)
