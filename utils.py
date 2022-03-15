import os
import time
import shutil
from collections import OrderedDict
from easydict import EasyDict
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# No need to write model definition, RobustBench and TorchVision already hosts
# widely used network architectures in robust ML.
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.architectures.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
)
from robustbench.model_zoo.architectures.resnest import ResNest152
import torchvision.models as tv_models
from autoattack import AutoAttack


class CustomResNet(ResNet):
    """
    Replacing avg_pool with a adaptive_avg_pool. Now this model can be used much
    resolution beyond cifar10.
    Note: ResNet models in RobustBench are cifar10 style, thus geared to 32x32. These
    models are slightly different than original ResNets (224x224 resolution).
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(CustomResNet, self).__init__(block, num_blocks, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_model(arch, num_classes):
    # we use lower case letter for cifar10 (i.e., for 32x32 and 64x64 images) style models and
    # upper case letter, such as ResNet18, for ImageNet (224x224 size images) style models.

    if arch in ["wrn_28_1", "wrn_28_10", "wrn_34_10", "wrn_70_16"]:
        model = WideResNet(
            depth=int(arch.split("_")[-2]),
            widen_factor=int(arch.split("_")[-1]),
            num_classes=num_classes,
        )
    elif arch in ["resnet18", "resnet50"]:
        if arch == "resnet18":
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        if arch == "resnet50":
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif arch in ["resnet18_64", "resnet50_64"]:
        if arch == "resnet18_64":
            model = CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        if arch == "resnet50_64":
            model = CustomResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif arch in ["ResNet18", "ResNet50"]:
        model = tv_models.__dict__[arch.lower()](
            num_classes=num_classes, pretrained=False
        )
    elif arch == "resnest152":
        model = ResNest152(num_classes=num_classes)
    else:
        raise ValueError(
            f"{arch} is not imported! Please import it from robustbench or torchvision model zoo and update this function accordingly."
        )
    return model


def get_metadata(dataset):
    metadata = {
        "cifar10": {
            "num_classes": 10,
            "image_size": 32,
            "train_images": 50000,
            "val_images": 10000,
        },
        "cifar100": {
            "num_classes": 10,
            "image_size": 32,
            "train_images": 50000,
            "val_images": 10000,
        },
        "imagenet64": {
            "num_classes": 1000,
            "image_size": 224,
            "train_images": 1281167,
            "val_images": 50000,
        },
        "celebA": {
            "num_classes": 4,
            "image_size": 64,
            "train_images": 109036,
            "val_images": 12376,
        },
    }

    assert dataset in metadata.keys(), f"metdata not available for {dataset} dataset."
    return EasyDict(metadata[dataset])


def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_dir, filename),
            os.path.join(save_dir, "model_best.pth.tar"),
        )


class combine_dataloaders:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2

    def __iter__(self):
        return self._iterator()

    def __len__(self):
        return min(len(self.dataloader1), len(self.dataloader2))

    def _iterator(self):
        for (img1, label1), (img2, label2) in zip(self.dataloader1, self.dataloader2):
            images = torch.cat([img1, img2])
            labels = torch.cat([label1, label2])
            indices = torch.randperm(len(images))
            yield images[indices], labels[indices]


# https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d


def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def trainfxn(
    trainer,
    model,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    num_classes,
    logger,
    **kwargs,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    if num_classes >= 5:
        top5 = AverageMeter("Acc@5", ":6.2f")
    else:
        top5 = AverageMeter("Acc@2", ":6.2f")  # measuring top-2
    if trainer in ["pgd", "fgsm", "trades"]:
        top1_adv = AverageMeter("AccAdv@1", ":6.2f")
        if num_classes >= 5:
            top5_adv = AverageMeter("AccAdv@5", ":6.2f")
        else:
            top5_adv = AverageMeter("AccAdv@2", ":6.2f")  # measuring top-2
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, losses, top1, top5, top1_adv, top5_adv],
            prefix="Epoch: [{}]".format(epoch),
        )
    elif trainer == "baseline":
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )
    else:
        raise ValueError(f"trainer {trainer} not supported")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(dataloader):
        data_time.update(time.time() - end)

        # basic properties
        if i == 0 and args.rank == 0:
            logger.info(
                f"Batch images shape: {images.shape}, targets shape: {targets.shape}, "
                + f"World-size: {args.world_size}, "
                f"Effective batch size: {args.world_size * len(images)}, "
                + f"Learning rate (epoch {epoch}/{args.epochs}): {optimizer.param_groups[0]['lr']:.5f}, "
                + f"pixel range: {[images.min().item(), images.max().item()]}"
            )
        images, targets = images.cuda(args.gpu, non_blocking=True), targets.cuda(
            args.gpu, non_blocking=True
        )
        logits = model(images)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5 if num_classes >= 5 else 2))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if trainer in ["fgsm", "pgd", "trades"]:
            logits_adv, loss = get_adversarial_loss(
                trainer, model, images, targets, logits, criterion, optimizer, args
            )
            acc1_adv, acc5_adv = accuracy(
                logits_adv, targets, topk=(1, 5 if num_classes >= 5 else 2)
            )
            top1_adv.update(acc1_adv[0], images.size(0))
            top5_adv.update(acc5_adv[0], images.size(0))
        elif trainer in ["baseline"]:
            loss = criterion(logits, targets)

        losses.update(loss.item(), images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)

    if trainer in ["fgsm", "pgd", "trades"]:
        result = {
            "top1": top1.avg,
            f"top{5 if num_classes >= 5 else 2}": top5.avg,
            "top1_adv": top1_adv.avg,
            f"top{5 if num_classes >= 5 else 2}_adv": top5_adv.avg,
        }
    elif trainer in ["baseline"]:
        result = {"top1": top1.avg, f"top{5 if num_classes >= 5 else 2}": top5.avg}
    return result


def evalfxn(
    val_method,
    model,
    dataloader,
    criterion,
    args,
    num_classes,
    **kwargs,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    if num_classes >= 5:
        top5 = AverageMeter("Acc@5", ":6.2f")
    else:
        top5 = AverageMeter("Acc@2", ":6.2f")  # measuring top-2
    if val_method in ["pgd", "auto"]:
        top1_adv = AverageMeter("AccAdv@1", ":6.2f")
        if num_classes >= 5:
            top5_adv = AverageMeter("AccAdv@5", ":6.2f")
        else:
            top5_adv = AverageMeter("AccAdv@2", ":6.2f")  # measuring top-2
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, losses, top1, top5, top1_adv, top5_adv],
            prefix="Test: ",
        )
    elif val_method == "baseline":
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Test: ",
        )
    else:
        raise ValueError(f"Trainer {val_method} not supported")

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, targets) in enumerate(dataloader):
        data_time.update(time.time() - end)
        images, targets = images.cuda(args.gpu, non_blocking=True), targets.cuda(
            args.gpu, non_blocking=True
        )

        logits = model(images)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5 if num_classes >= 5 else 2))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if val_method in ["pgd", "auto"]:
            logits_adv, loss = get_adversarial_loss(
                val_method, model, images, targets, logits, criterion, None, args
            )
            acc1_adv, acc5_adv = accuracy(
                logits_adv, targets, topk=(1, 5 if num_classes >= 5 else 2)
            )
            top1_adv.update(acc1_adv[0], images.size(0))
            top5_adv.update(acc5_adv[0], images.size(0))
        elif val_method in ["baseline"]:
            loss = criterion(logits, targets)
            
        losses.update(loss.item(), images.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)

    if val_method in ["pgd", "auto"]:
        result = {
            "top1": top1.avg,
            f"top{5 if num_classes >= 5 else 2}": top5.avg,
            "top1_adv": top1_adv.avg,
            f"top{5 if num_classes >= 5 else 2}_adv": top5_adv.avg,
        }
    elif val_method in ["baseline"]:
        result = {"top1": top1.avg, f"top{5 if num_classes >= 5 else 2}": top5.avg}
    else:
        ValueError(f"{val_method} validation method not supported!")
    return result


def pgd_attack(
    model,
    criterion,
    images,
    targets,
    epsilon,
    step_size,
    num_steps,
    attack,
    clip_min,
    clip_max,
):
    images, targets = images.detach(), targets.detach()
    if attack == "linf":
        with torch.enable_grad():
            eps = torch.nn.Parameter(
                torch.zeros_like(images).uniform_(-epsilon, epsilon), requires_grad=True
            )
            for i in range(num_steps):
                logits_adv = model(images + eps)
                loss_adv = criterion(logits_adv, targets)
                grad = torch.autograd.grad(loss_adv, eps, create_graph=False)[0]
                eps.data = eps.data + step_size * grad.sign()
                eps.data = (images + eps.data.clamp(-epsilon, epsilon)).clamp(
                    clip_min, clip_max
                ) - images
        adv_images = images + eps.detach()
    else:
        raise ValueError("Attack not supported")
    return adv_images


# Ref: https://github.com/yaodongyu/TRADES/blob/master/trades.py
# Removed redundant forward passes (~1.2x speedup)
def trades_loss(
    model,
    x_natural,
    y,
    logits_natural,
    optimizer,
    epsilon=0.031,
    step_size=0.003,
    perturb_steps=10,
    clip_min=0.0,
    clip_max=1.0,
    beta=1.0,
    distance="linf",
):
    # define KL-loss
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == "linf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(logits_natural.detach(), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l2":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = torch.optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1),
                    F.softmax(logits_natural.detach(), dim=1),
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return logits_adv, loss


def get_adversarial_loss(
    trainer, model, images, targets, logits, criterion, optimizer, args
):
    if trainer == "pgd":
        adv_images = pgd_attack(
            model,
            criterion,
            images,
            targets,
            args.epsilon,
            args.step_size,
            args.num_steps,
            args.attack,
            args.clip_min,
            args.clip_max,
        )
        logits_adv = model(adv_images)
        loss_adv = criterion(logits_adv, targets)
    elif trainer == "trades":
        logits_adv, loss_adv = trades_loss(
            model,
            images,
            targets,
            logits,
            optimizer,
            args.epsilon,
            args.step_size,
            args.num_steps,
            args.clip_min,
            args.clip_max,
            6.0,
            args.attack,
        )
    elif trainer == "auto":
        adversary = AutoAttack(
            model, norm="Linf" if args.attack == "linf" else "L2", eps=args.epsilon
        )
        adversary.attacks_to_run = ["apgd-ce", "apgd-t"]
        adv_images = adversary.run_standard_evaluation(images, targets, bs=len(images))
        logits_adv = model(adv_images)
        loss_adv = criterion(logits_adv, targets)
    else:
        raise ValueError(f"{trainer} attack is not supported")
    return logits_adv, loss_adv
