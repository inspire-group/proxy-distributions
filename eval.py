import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils import (
    get_model,
    get_metadata,
    fix_legacy_dict,
    evalfxn,
)
from data import get_real_dataloaders


def main():
    parser = argparse.ArgumentParser("Evalution script")

    # common args
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument(
        "--results-dir", type=str, default="./trained_models/eval_logs/"
    )
    parser.add_argument(
        "--val-method",
        type=str,
        default="pgd",
        choices=("baseline", "pgd", "auto"),
    )

    # data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10", "cifar100", "imagnet64", "celebA", "afhq"),
    )
    parser.add_argument("--data-dir", type=str, help="dir where data is stored")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)

    # model
    parser.add_argument("--arch", type=str)

    # adversarial attack
    parser.add_argument("--attack", type=str, choices=("linf", "l2"), default="linf")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255)
    parser.add_argument("--step-size", type=float, default=2.0 / 255)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    parser.add_argument(
        "--autoattack-attack-subset",
        nargs="+",
        help="subset of attacks from autoattack default set of attacks (default uses all attacks)",
    )

    # misc
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)

    ######################### Basic setup #########################
    args = parser.parse_args()
    args.metadata = get_metadata(args.dataset)
    args.gpu, args.rank = "cuda:0", 0  # now its compatible with training utils
    print(args)

    # set up cuda + seeds
    torch.backends.cudnn.benchmark = True  # a few percentage speedup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    assert (
        torch.cuda.is_available()
    ), "we assume cuda is available, too slow to eval on cpus."
    assert args.checkpoint_path, "Need a checkpoint to evaluate"

    # recursively create all directories needed to log results
    args.log_dir = os.path.join(
        os.path.join(args.results_dir, args.exp_name), f"trial_{args.trial}"
    )
    os.makedirs(args.log_dir, exist_ok=True)

    # logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(args.log_dir, "eval_log.txt"), "a")
    )
    logger.info(args)

    model = get_model(args.arch, args.metadata.num_classes).cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    d = fix_legacy_dict(checkpoint["state_dict"])
    logger.info(f"Loaded state dict from {args.checkpoint_path}")
    model.load_state_dict(d, strict=True)
    logger.info(f"Mismatched keys {set(d.keys()) ^ set(model.state_dict().keys())}")
    logger.info(f"Checkpoint loaded from {args.checkpoint_path}")

    model = torch.nn.parallel.DataParallel(model).eval()
    criterion = nn.CrossEntropyLoss().cuda()

    _, val_loader, _, _ = get_real_dataloaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.workers,
        args.metadata,
        distributed=False,
    )
    results_val = evalfxn(
        args.val_method,
        model,
        val_loader,
        criterion,
        args,
        args.metadata.num_classes,
    )
    logger.info(
        ", ".join(
            ["{}: {:.3f}".format(k + "_val", v) for (k, v) in results_val.items()]
        )
    )


if __name__ == "__main__":
    main()
