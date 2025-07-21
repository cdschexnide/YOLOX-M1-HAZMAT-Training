#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX Hazmat Detection Configuration - Optimized for Google Colab GPU Training

import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model config for YOLOX-S
        self.num_classes = 12
        self.depth = 0.33
        self.width = 0.50
        
        # All 12 hazmat class names found in annotations
        self.class_names = [
            "corrosive", "dangerous-when-wet", "explosive", "flammable", 
            "flammable-solid", "infectious-substance", "non-flammable-gas", 
            "organic-peroxide", "oxidizer", "poison", "radioactive", 
            "spontaneously-combustible"
        ]
        
        # Training config optimized for Google Colab T4 GPU (16GB VRAM)
        self.max_epoch = 25  # Reduced for proof of concept
        self.warmup_epochs = 3
        self.data_num_workers = 2  # Colab has limited CPU cores
        self.eval_interval = 5
        
        # Colab GPU optimizations
        self.fp16 = True  # Enable mixed precision for GPU training
        
        # Colab-specific paths (will be mounted from Google Drive)
        self.dataset_root = "/content/drive/MyDrive/hazmat_dataset"
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler, 
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master, get_local_rank
        from yolox.data.datasets.voc import AnnotationTransform
        
        local_rank = get_local_rank()
        
        # Create hazmat class mapping
        hazmat_class_to_ind = dict(zip(self.class_names, range(len(self.class_names))))

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(
                data_dir=os.path.join(self.dataset_root, "VOCdevkit"),
                image_sets=[('2007', 'train')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                ),
                target_transform=AnnotationTransform(class_to_ind=hazmat_class_to_ind),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import VOCDetection, ValTransform
        from yolox.data.datasets.voc import AnnotationTransform

        # Create hazmat class mapping
        hazmat_class_to_ind = dict(zip(self.class_names, range(len(self.class_names))))

        valdataset = VOCDetection(
            data_dir=os.path.join(self.dataset_root, "VOCdevkit"),
            image_sets=[('2007', 'val')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            target_transform=AnnotationTransform(class_to_ind=hazmat_class_to_ind),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.get_model().named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, torch.nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})

            self.optimizer = optimizer

        return self.optimizer