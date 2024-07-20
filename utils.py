import itertools
import random
from pathlib import Path
from typing import Tuple, Iterable, Optional, Dict, List

import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch import Tensor
import numpy as np

import torchvision.utils as vutils


def get_label_to_dev_samples(
        mnist_dev_loader: DataLoader,
        num_samples_per_class: int = 10,
        num_classes: int = 10
) -> Dict[int, List[Tensor]]:
    label_to_samples = {label_idx: [ ] for label_idx in range(num_classes)}
    total_num_samples = num_samples_per_class * num_classes
    num_added_samples = 0
    while num_added_samples < total_num_samples:
        for data_batch, label_batch in mnist_dev_loader:
            for data_sample, label_sample in zip(data_batch, label_batch):
                if len(label_to_samples[int(label_sample)]) < num_samples_per_class:
                    label_to_samples[int(label_sample)].append(data_sample)
                    num_added_samples += 1

    return label_to_samples



def get_mnist_transforms(flatten: bool = False) -> transforms.Compose:
    transforms_list = [
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ]
    if flatten:
        transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

    return transforms.Compose(transforms_list)


def get_mnist_train_dev_loaders(
        mnist_root_dir: Path,
        batch_size: int,
        flatten_img: bool = False

):
    transforms = get_mnist_transforms(flatten_img)
    mnist_path_str = str(mnist_root_dir.absolute())
    train_set = torchvision.datasets.MNIST(
        mnist_path_str,
        train=True,
        transform=transforms,
        download=True
    )
    dev_set = torchvision.datasets.MNIST(
        mnist_path_str,
        train=False,
        transform=transforms,
        download=True
    )
    return (
        DataLoader(train_set, batch_size, shuffle=True),
        DataLoader(dev_set, batch_size, shuffle=True)
    )


def get_cifar10_train_dev_loaders(
        cifar_root_dir: Path,
        batch_size: int,
        dev_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    # Transform a single image - Convert it to a tensor and normalize it
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the train set of CIFAR10 (there is only train and test)
    cifar_train = CIFAR10(cifar_root_dir, train=True, download=True, transform=transform)

    # Create a list of the indices all data samples
    num_total_samples = len(cifar_train)
    all_cifar_sample_idx = list(range(num_total_samples))
    # Shuffle the list for random indecing
    random.shuffle(all_cifar_sample_idx)

    # Split the list of indices into train and dev sets
    train_ratio = 1.0 - dev_ratio
    num_train_samples = int(train_ratio * num_total_samples)
    train_indices = all_cifar_sample_idx[:num_train_samples]
    dev_indices = all_cifar_sample_idx[num_train_samples:]

    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)

    train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True,
                                                    sampler=dev_sampler)

    return train_loader, validation_loader


def write_real_example_images_tensorboard(
        data_loader: DataLoader,
        tb_writer: SummaryWriter,
        global_step: int,
        tb_tag: str = "real_images",
        num_examples_per_class: int = 2,
        img_shape: int = (1, 28, 28),
        num_classes: int = 10
):
    # Get num_examples_per_class samples for each class
    class_to_tensors = get_label_to_dev_samples(data_loader, num_examples_per_class, num_classes)
    # Flatten the dict of classes to samples to a list
    example_samples_flat = list(itertools.chain.from_iterable(class_to_tensors[class_idx] for class_idx in range(num_classes)))
    samples_tensor = torch.stack(example_samples_flat)
    samples_tensor = samples_tensor.reshape(num_classes*num_examples_per_class, *img_shape)
    tensor_grid = vutils.make_grid(samples_tensor, normalize=True, nrow=num_classes)
    tb_writer.add_image(tb_tag, tensor_grid, global_step)


def write_fake_gan_example_images_tensorboard(
        gan,
        tb_writer: SummaryWriter,
        global_step: int,
        tb_tag: str = "fake_images",
        num_samples: int = 20,
        img_shape: int = (1, 28, 28),
        device: torch.device = torch.device("cpu")
):
    samples = gan.create_new_samples(num_samples, device)
    samples = samples.reshape(num_samples, *img_shape)
    tensor_grid = vutils.make_grid(samples, normalize=True, nrow=10)
    tb_writer.add_image(tb_tag, tensor_grid, global_step)
