import sys
import os.path as osp

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from loader_five import get_loader as get_five_loader
import loader_pmnist
from loader_cifar100 import get_train_valid_loader_cifar100, get_test_loader_cifar100

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="data")
args = parser.parse_args()

# Prepare datasets
print("---> Preparing Five datasets ...")
train_loader_splits, valid_loader_splits, task_output_space = get_five_loader(
    seed=0,
    loader_type="train",
    fixed_order=False,
    batch_size=32,
    pc_valid=0.1,
    num_workers=1,
    pin_memory=True,
)
test_loader_splits = get_five_loader(
    seed=0,
    loader_type="test",
    batch_size=32,
    pc_valid=0.1,
    num_workers=1,
    pin_memory=True,
)

print("---> Preparing pmnist datasets ...")
(
    train_loader_splits,
    valid_loader_splits,
    task_output_space,
) = loader_pmnist.get_loader(
    seed=0,
    loader_type="train",
    batch_size=32,
    fixed_order=False,
    pc_valid=0.1,
    num_workers=1,
    pin_memory=True,
)

test_loader_splits = loader_pmnist.get_loader(
    seed=0,
    loader_type="test",
    batch_size=32,
    pc_valid=0.1,
    num_workers=1,
    pin_memory=True,
)

print("---> Preparing cifar100 datasets ...")
n_task = 10
n_class = 100
# Split data into tasks
step_size = int(n_class / n_task)
dict_classSplit = {}
for i in range(0, n_task):
    dict_classSplit[i] = list(range(i * step_size, (i + 1) * step_size))
task_order_dict = {
    "A": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "B": [1, 7, 4, 5, 2, 0, 8, 6, 9, 3],
    "C": [7, 0, 5, 1, 8, 4, 3, 6, 2, 9],
    "D": [5, 8, 2, 9, 0, 4, 3, 7, 6, 1],
    "E": [2, 9, 5, 4, 8, 0, 6, 1, 3, 7],
}
(
    train_loader_splits,
    valid_loader_splits,
    task_output_space,
) = get_train_valid_loader_cifar100(
    args.data_root,
    32,
    dict_classSplit,
    random_seed=111111,
    train_size=1.0,
    valid_size=0.1,
    augment=True,
    num_workers=1,
    pin_memory=True,
)

test_loader_splits = get_test_loader_cifar100(
    args.data_root,
    32,
    dict_classSplit,
    num_workers=1,
    pin_memory=True,
)
