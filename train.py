# A minimal script for multi-node multi-gpu adversarial training.
# Modelled after official pytorch example script from https://github.com/pytorch/examples/blob/main/imagenet/main.py

import argparse
import os
import numpy as np
import warnings
import logging
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils import (
    get_model,
    get_metadata,
    combine_dataloaders,
    save_checkpoint,
    trainfxn,
    evalfxn,
)
from data import get_real_dataloaders, get_synthetic_dataloaders

best_prec = 0

def main():
    parser = argparse.ArgumentParser("Robust training with proxy distributions")

    # common args
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument("--results-dir", type=str, default="./trained_models/")

    # data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10", "cifar100"),
    )
    parser.add_argument("--data-dir", type=str, default="./datasets/", help="dir where data is stored")

    # synthetic data
    parser.add_argument(
        "--syn-data-list",
        nargs="+",
        choices=("ddpm_cifar10", "ddpm_cifar100"),
        help="list of all synthetic datasets to uses",
    )
    parser.add_argument(
        "--syn-data-dir",
        nargs="+",
        help="individual dir where data is stored for each synthetic dataset",
    )
    parser.add_argument(
        "--batch-size-syn", type=int, default=128, help="batch-size for synthetic data"
    )
    parser.add_argument("--gamma", type=float, default=0.5)

    # model
    parser.add_argument("--arch", type=str)
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        default=False,
        help="Synchronize batch-norm across all devices.",
    )

    # training details
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--trainer",
        type=str,
        default="pgd",
        choices=("baseline", "trades", "pgd", "fgsm"),
    )
    parser.add_argument(
        "--val-method",
        type=str,
        default="pgd",
        choices=("baseline", "pgd", "auto"),
    )

    # adversarial attack
    parser.add_argument("--attack", type=str, choices=("linf", "l2"), default="linf")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255)
    parser.add_argument("--step-size", type=float, default=2.0 / 255)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)

    # distributed training
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--no-dist",
        action="store_true",
        default=False,
        help="Don't use distributed training (E.g, set it to true for single-gpu)",
    )

    # misc
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)

    ######################### Basic setup #########################
    args = parser.parse_args()
    args.metadata = get_metadata(args.dataset)

    # set up cuda + seeds
    torch.backends.cudnn.benchmark = True  # ~20% speedup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    assert (
        torch.cuda.is_available()
    ), "we assume cuda is available, too slow to train on cpus."
    if args.syn_data_list or args.syn_data_dir:
        assert len(args.syn_data_list) == len(args.syn_data_dir)

    # recursively create all directories needed to log results
    args.checkpoint_dir = os.path.join(
        os.path.join(args.results_dir, args.exp_name), f"trial_{args.trial}"
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ######################### Multiprocessing #########################
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    if args.dist_url == "env://" and args.world_size == -1:
        if args.no_dist:
            args.world_size = 1
        else:
            args.world_size = int(os.environ["WORLD_SIZE"])
    args.multiprocessing_distributed = args.world_size > 1

    if args.multiprocessing_distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()

        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    else:
        args.rank = 0
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_prec
    args.gpu = gpu

    if args.rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger()
        logger.addHandler(
            logging.FileHandler(os.path.join(args.checkpoint_dir, "train_log.txt"), "a")
        )
        logger.info(args)
    else:
        logger = None

    model = get_model(args.arch, args.metadata.num_classes)
    if args.gpu is not None:
        # Single GPU per process -> divide batch size based by number of GPUs
        args.batch_size = int(args.batch_size / args.world_size)
        args.batch_size_syn = int(args.batch_size_syn / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)

        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # batchnorm will use local device batch data for normalization.
        # synchronized-batchnorm will pool data from all devices for normalization
        # its take ~1.4x longer to train  with syncBN but it makes BN stable
        # (for very small batch-size) 2) Slighly better convergence
        if args.sync_bn:
            print("Using synchronize batch-normalization!")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            assert args.batch_size >= 4, (
                f"Please use modestly large-batch size per GPU "
                + f"(current batch-size/GPU = {args.batch_size}), since only local stats "
                + "are used for batch-norm. Smaller batch-size makes BN unstable."
            )
        # Using Broadcast_buffers=False due to Batch-norm using mean/var buffers.
        # If true it broadcast buffers of rank 0 process to all others. Without it
        # this sync won't happen and each device will maintain its own running mean/var.
        # Doing so allows us to do more than one forward pass on Batch-norm
        # in train() mode (https://github.com/pytorch/pytorch/issues/22095)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], broadcast_buffers=False
        )
    else:
        # Use DistributedDataParallel since it is much faster than DataParallel
        # We added DataParallel just for the sake of it.
        model = torch.nn.DataParallel(model).cuda()

    ######################### Dataloaders #########################
    train_loader, val_loader, train_sampler, val_sampler = get_real_dataloaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.workers,
        args.metadata,
        distributed=False if args.no_dist else True,
    )
    if args.rank == 0:
        logger.info(
            "Using distributed sampler on validation data. This speed up validation process "
            + "(which matters when doing adversarial validation), but may end up being just a little bit inaccurate. "
            "MAKE SURE TO RUN THE FINAL EVALUATION WITH EVAL.PY. Remove val_sampler if you want "
            + "to do a correct but slow evaluation."
        )
    samplers = [train_sampler, val_sampler]
    if args.syn_data_list:
        if args.rank == 0:
            logger.info(
                f"Using following synthetic dataset in addition to real data: {args.syn_data_list}"
            )
        for s, d in zip(args.syn_data_list, args.syn_data_dir):
            syn_data_loader, syn_data_sampler = get_synthetic_dataloaders(
                syn_dataset=s,
                syn_data_dir=d,
                real_dataset=args.dataset,
                batch_size=args.batch_size_syn,
                num_workers=args.workers,
                metadata=args.metadata,
                distributed=False if args.no_dist else True,
            )  # using identical batch-size for all synthetic datasets in list
            samplers.append(syn_data_sampler)
            # One can easily manipulate this line to use only real or synthetic data
            # by default we combine both of them
            train_loader = combine_dataloaders(train_loader, syn_data_loader)
        if args.rank == 0:
            logger.info(
                f"Ratio of real to synthetic data {1}:{(len(args.syn_data_list)*args.batch_size_syn)/args.batch_size}"
            )

    ######################### Training setup #########################
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.rank == 0:
        logger.info(
            f"no. of batches = {len(train_loader)}, real data batch_size/GPU = {args.batch_size} "
            + f"synthetic data batch_size/GPU = {args.batch_size_syn}"
            if args.syn_data_list
            else "",
        )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), eta_min=0.001
    )

    ######################### Let's roll #########################
    for epoch in range(args.epochs):
        for sam in samplers:
            if sam is not None:
                sam.set_epoch(epoch)

        results_train = trainfxn(
            args.trainer,
            model,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            epoch,
            args,
            args.metadata.num_classes,
            logger,
        )
        results_val = evalfxn(
            args.val_method,
            model,
            val_loader,
            criterion,
            args,
            args.metadata.num_classes,
        )

        if args.rank == 0:
            if args.val_method == "baseline":
                prec = results_val["top1"]
            elif args.val_method in ["pgd", "trades", "fgsm"]:
                prec = results_val["top1_adv"]
            else:
                raise ValueError()
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)

            state_dict = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec,
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state_dict, is_best, save_dir=args.checkpoint_dir)

            logger.info(
                f"Epoch {epoch}, "
                + ", ".join(
                    [
                        "{}: {:.3f}".format(k + "_train", v)
                        for (k, v) in results_train.items()
                    ]
                    + [
                        "{}: {:.3f}".format(k + "_val", v)
                        for (k, v) in results_val.items()
                    ]
                )
            )


if __name__ == "__main__":
    main()
