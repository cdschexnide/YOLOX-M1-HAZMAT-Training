#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # Model parameters
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Hazmat-specific parameters
        self.num_classes = 9
        self.class_names = [
            "explosive",
            "flammable", 
            "oxidizer",
            "dangerous-when-wet",
            "poison",
            "spontaneously-combustible",
            "radioactive",
            "corrosive",
            "non-flammable-gas"
        ]

        # Training parameters optimized for M1
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 5
        self.warmup_epochs = 5

        # Learning rate
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        # Weight decay and momentum
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # Augmentation parameters
        self.enable_mixup = True
        self.mixup_prob = 0.5
        self.mosaic_prob = 0.8
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mixup_scale = (0.5, 1.5)
        self.mosaic_scale = (0.1, 2)
        self.shear = 2.0

        # Input/Output
        self.input_size = (640, 640)
        self.multiscale_range = 5
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # M1-specific settings
        self.fp16 = False

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
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        from yolox.data.datasets.voc import AnnotationTransform
        
        local_rank = get_local_rank()

        # Create custom class mapping for hazmat classes
        hazmat_class_to_ind = dict(zip(self.class_names, range(len(self.class_names))))

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(
                data_dir="datasets/hazmatVOC/VOCdevkit",
                image_sets=[('2007', 'train')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
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
                hsv_prob=self.hsv_prob),
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

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

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

        # Create custom class mapping for hazmat classes
        hazmat_class_to_ind = dict(zip(self.class_names, range(len(self.class_names))))

        valdataset = VOCDetection(
            data_dir="datasets/hazmatVOC/VOCdevkit", 
            image_sets=[('2007', 'val')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            target_transform=AnnotationTransform(class_to_ind=hazmat_class_to_ind),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
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