#!/usr/bin/env python3
import argparse
import os
from yolox.core import launch
from yolox.exp import get_exp

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Hazmat Training on M1 Mac")
    parser.add_argument("-expn", "--experiment-name", type=str,
default="yolox_s_hazmat_m1")
    parser.add_argument("-n", "--name", type=str,
default="yolox_s_hazmat_m1", help="model name")
    parser.add_argument("-f", "--exp_file",
default="exps/hazmat/yolox_s_hazmat_m1.py", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=8,
help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int,
help="device for training")
    parser.add_argument("-c", "--ckpt", default=None, type=str,
help="checkpoint file")
    parser.add_argument("--resume", default=False, action="store_true",
help="resume training")
    parser.add_argument("--fp16", dest="fp16", default=False,
action="store_true", help="Mixed precision")
    parser.add_argument("--cache", dest="cache", default=False,
action="store_true", help="Cache images")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False,
action="store_true", help="occupy memory")
    parser.add_argument("--logger", default="tensorboard", type=str,
help="logger type")
    return parser

def main(exp, args):
    # Setup M1 environment
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Get trainer and start training
    trainer = exp.get_trainer(args)
    trainer.train()

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    # Update experiment with args values
    if args.batch_size:
        exp.batch_size = args.batch_size

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # Single device training for M1
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        backend="gloo",
        args=(exp, args),
    )