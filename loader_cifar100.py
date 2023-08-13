# -*- coding: utf-8 -*-
"""

[1]: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader_cifar100(
    dataroot,
    batch_size,
    dict_classSplit,
    random_seed,
    train_size=1.0,
    valid_size=0.2,
    augment=False,
    num_workers=4,
    pin_memory=False,
    shuffle=False,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataroot: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - dict_classSplit: dictionary indicating how classes are split into non-overlapped group
    - random_seed: fix seed for reproducibility (shuffle the train/validation indices).
    - train_size: percentage of training data used as a subset for training
    - valid_size: percentage split of the training (sub)set used for
      the validation set. Should be a float in the range [0, 1].
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader_dict: dictionary including training set iterator for each task.
    - valid_loader_dict: dictionary including validation set iterator for each task
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (train_size >= 0) and (train_size <= 1), error_msg
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
    )

    # define transforms
    valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = valid_transform

    # load the dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot, train=True, download=True, transform=train_transform
    )

    valid_dataset = torchvision.datasets.CIFAR100(
        root=dataroot, train=True, download=True, transform=valid_transform
    )

    # Split dataset into groups of non-overlapped classes
    indices_splits = {}
    task_output_space = {}
    label_raw = np.array(train_dataset.targets.copy())
    label_remap = np.array(train_dataset.targets.copy())
    for name, class_list in dict_classSplit.items():
        indices_splits[name] = [np.where(label_raw == c)[0] for c in class_list]
        task_output_space[name] = len(class_list)
        for i, c in enumerate(class_list):
            label_remap[indices_splits[name][i]] = i

        indices_splits[name] = np.hstack(indices_splits[name])

    train_dataset.targets = label_remap.tolist()
    valid_dataset.targets = label_remap.tolist()

    # Create dataloader for each task
    train_loader_splits = {}
    valid_loader_splits = {}
    np.random.seed(random_seed)

    for name in dict_classSplit.keys():
        num_task_size = len(indices_splits[name])
        num_task_subsize = int(np.ceil(train_size * num_task_size))
        val_task_split = int(np.floor(valid_size * num_task_subsize))
        indices = indices_splits[name].tolist()
        np.random.shuffle(indices)
        indices = indices[:num_task_subsize]  # take a subset of data
        train_idx, valid_idx = indices[val_task_split:], indices[:val_task_split]

        train_loader_splits[name] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, train_idx),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        valid_loader_splits[name] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(valid_dataset, valid_idx),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return (train_loader_splits, valid_loader_splits, task_output_space)


def get_test_loader_cifar100(
    dataroot, batch_size, dict_classSplit, num_workers=4, pin_memory=False
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataroot: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - dict_classSplit: dictionary indicating how classes are split into non-overlapped group
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - test_loader_dict: dictionary including test set iterator for each task.
    """

    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
    )

    # define transforms
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # load the dataset
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataroot, train=False, download=False, transform=test_transform
    )

    # Split dataset into groups of non-overlapped classes
    indices_splits = {}
    label_raw = np.array(test_dataset.targets.copy())
    label_remap = np.array(test_dataset.targets.copy())
    for name, class_list in dict_classSplit.items():
        indices_splits[name] = [np.where(label_raw == c)[0] for c in class_list]
        for i, c in enumerate(class_list):
            label_remap[indices_splits[name][i]] = i

        indices_splits[name] = np.hstack(indices_splits[name])

    test_dataset.targets = label_remap.tolist()

    # Create dataloader for each task
    test_loader_splits = {}
    for name in dict_classSplit.keys():
        indices = indices_splits[name].tolist()
        test_loader_splits[name] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(indices),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return test_loader_splits
