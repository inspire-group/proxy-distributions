import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

TRANFORMS_MAPPING = {
    "cifar10": "cifar_style",
    "cifar100": "cifar_style",
    "celebA": "cifar_style",
    "imagenet64": "cifar_style",
    "afhq": "imagenet_fixed_size_style",
}


def get_transforms(name, image_size):
    if name == "cifar_style":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=int(image_size / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_val = transforms.ToTensor()
    elif name == "imagenet_style":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.Resize(int(1.14 * image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
    elif name == "imagenet_fixed_size_style":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        # assuming all images are resized to a fixed resolution (e.g., 224x224)
        # which makes resizing unnecessary for validation set
        transform_val = transforms.ToTensor()
    else:
        raise ValueError(f"{name} data transformation not supported!")
    return transform_train, transform_val


def get_dataset(dataset, data_dir, transform_train, transform_val, distributed):
    if dataset == "cifar10":
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        val_set = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_val,
        )
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        val_set = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_val,
        )
    elif dataset == "celebA":
        train_set = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=transform_train
        )
        val_set = datasets.ImageFolder(
            os.path.join(data_dir, "val"), transform=transform_val
        )
    else:
        raise ValueError(f"{dataset} dataset not supported!")

    return (
        train_set,
        val_set,
        DistributedSampler(train_set) if distributed else None,
        DistributedSampler(val_set) if distributed else None,
    )


def get_dataloader(dset, batch_size, num_workers, sampler=None):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_real_dataloaders(
    dataset, data_dir, batch_size, num_workers, metadata, distributed=True
):
    transform_train, transform_val = get_transforms(
        TRANFORMS_MAPPING[dataset], metadata.image_size
    )
    train_set, val_set, train_sampler, val_sampler = get_dataset(
        dataset, data_dir, transform_train, transform_val, distributed
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Ref: https://github.com/pytorch/examples/issues/769
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler, val_sampler


class SyntheticImagesDdpmCIFAR10(torch.utils.data.Dataset):
    """
    Load synthetic images (6002688 from DDPM and 9400000 images from Improved-DDPM model)
    """

    def __init__(self, src, labels):
        """
        src: .bin file of data
        labels: .numpy file of labels (in exact same order as src images)
        """
        self.src = src
        self.labels = np.load(labels)
        self.nddpm, self.nIddpm = 6002688, 9400000

    def sample_image(self, df, idx):
        df.seek(idx * 3072)
        image = np.array(np.frombuffer(df.read(3072), dtype="uint8").reshape(32, 32, 3))
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def __len__(self):
        return 15402688

    def __getitem__(self, idx):
        # opening src at initializing only leads to a race condition between workers in df.seek().
        # Opening it here avoids the race condition.
        with open(self.src, "rb") as df:
            img = self.sample_image(df, idx)
            df.close()
        label = self.labels[idx]
        return img, label


def get_synthetic_dataset(dataset, data_dir, transform_train, distributed):
    # TODO: Clean datasets such that we can simplify dataloading.add()
    # i.e., make all have same structure: data-dir => classes => images
    if dataset == "ddpm_cifar10":
        train_set = SyntheticImagesDdpmCIFAR10(
            os.path.join(data_dir, "cifar_ddpm_improvedddpm_sorted_images.bin"),
            os.path.join(data_dir, "cifar_ddpm_improvedddpm_sorted_labels.npy"),
        )
    elif dataset in ["ddpm_cifar100", "ddpm_celebA"]:
        train_set = datasets.ImageFolder(data_dir, transform=transform_train)
    else:
        raise ValueError(f"{dataset} dataset not supported!")

    return (
        train_set,
        DistributedSampler(train_set) if distributed else None,
    )


def get_synthetic_dataloaders(
    syn_dataset,
    syn_data_dir,
    real_dataset,
    batch_size,
    num_workers,
    metadata,
    distributed=True,
):
    transform_train, _ = get_transforms(
        TRANFORMS_MAPPING[real_dataset], metadata.image_size
    )
    train_set, train_sampler = get_synthetic_dataset(
        syn_dataset, syn_data_dir, transform_train, distributed
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, train_sampler
