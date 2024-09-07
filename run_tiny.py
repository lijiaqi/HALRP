import argparse
import os
import sys
import torch
import numpy as np
import pickle
import gc
from datetime import datetime
from utils.IOfun import Tee
import agents
from utils.metric import Timer, np2word
from utils.seralization import mkdir_if_missing

from loader_tiny.wrapper_task_inc import get_loader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpuid",
    nargs="+",
    type=int,
    default=[0],
    help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123551,
    help="Fix seed for reproducibility (generate random seed for train/val",
)
parser.add_argument("--repeat", type=int, default=1, help="Repeat the experiment N times")
parser.add_argument("--workers", type=int, default=0, help="#Thread for dataloader")
# parser.add_argument('--order_type', type=str, default='A', choices=['A', 'B','C','D','E'])
parser.add_argument("--fixed_order", type=bool, default=False)

parser.add_argument(
    "--model", type=str, default="AlexNet", choices=["AlexNet", "TinyNet", "ResNet18", "ResNet34"]
)

parser.add_argument(
    "--source_dir", type=str, default=".", help="The root folder of scripts"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data",
    help="The root folder of dataset or downloaded data",
)

parser.add_argument(
    "--method_desc",
    type=str,
    default="",
    help="Method description added to the file prefix",
)
parser.add_argument("--folderName", type=str, default="pmnist")
parser.add_argument(
    "--log_dir", type=str, default="logs", help="dir for saving model logs"
)
parser.add_argument(
    "--tab_dir",
    type=str,
    default="accTable",
    help="dir for saving model performance table",
)
parser.add_argument(
    "--tune_dir", type=str, default="tune", help="Temporal dir for saving best model"
)

parser.add_argument(
    "--train_aug",
    dest="train_aug",
    default=False,
    action="store_true",
    help="Allow data augmentation during training",
)
parser.add_argument(
    "--train_size",
    type=float,
    default=1.0,
    help="percentage of training data used as a subset for training",
)
parser.add_argument(
    "--valid_size",
    dest="valid_size",
    type=float,
    default=0.2,
    help="percentage split of the training (sub)set used for the validation set",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="SGD",
    help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...",
)
parser.add_argument("--n_epochs", type=int, default=5, help="Maximum number of epochs")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--schedule_gamma", type=float, default=1.0)
parser.add_argument(
    "--check_lepoch",
    type=int,
    default=3,
    help="The number from the last epoch for saving checkpoint",
)

parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument(
    "--print_freq", type=float, default=100, help="Print the log at every x mini-batches"
)
parser.add_argument(
    "--epoch_freq", type=float, default=1, help="Print the log at every x epochs"
)
parser.add_argument(
    "--agent_type", type=str, default="default", help="The type (filename) of agent"
)
parser.add_argument(
    "--agent_name", type=str, default="NormalNN", help="The class name of agent"
)
parser.add_argument("--stoptask_prop", type=float, default=1.0)
parser.add_argument("--rankMethod", type=str, default="imp")
parser.add_argument("--estRank_epoch", type=int, default=1)
parser.add_argument("--approxiRate", type=float, default=0.6)
parser.add_argument(
    "--upper_rank",
    type=int,
    default=1000,
    help="Upper bound of rank for perturbation to previous tasks",
)

# Regularizatoin
parser.add_argument("--l1_hyp", nargs="+", type=float, default=[0.0])
parser.add_argument(
    "--reg_coef",
    nargs="+",
    type=float,
    default=[0.0],
    help="The coefficient for loss approximation regularization.\
                        Larger means less plasilicity. Give a list for hyperparameter search.",
)
parser.add_argument("--wd_rate", type=float, default=1e-4)

# Weight pruning
parser.add_argument(
    "--prune_method",
    type=str,
    default="absolute",
    help="Threshold of pruning weights, \
                        can be absolute or relative or minAbsRel, default is absolute",
)
parser.add_argument("--prune_value", nargs="+", type=float, default=1e-4)
parser.add_argument(
    "--prune_boundAlltask",
    default=False,
    action="store_true",
    help="whether to turn the pruning rate to the upper bound of the final model.",
)
parser.add_argument(
    "--img_sz",
    type=int,
    default=64,
    help="image size is 64 for tiny imagenet dataset",
)
################
# for tiny-imagenet dataloader
################
parser.add_argument("--dataset", type=str, default="tinyimagenet", metavar="", help="")
parser.add_argument(
    "--loader_type",
    type=str,
    default="class_incremental_loader",
    choices=["class_incremental_loader", "task_incremental_loader"],
    help="incremental loader type",
)
parser.add_argument("--increment", type=int, default=5, metavar="S", help="(default: 5)")
parser.add_argument(
    "--class_order", type=str, default="random", help="tiny-imagenet order"
)
parser.add_argument("--pin_memory", type=bool, default=True)

parser.add_argument("--sparsity", type=float, default=0.1, help="sparsity 'c' in WSN")
######################################
# to define number of tasks for TinyImageNet
######################################
parser.add_argument("--num_tasks", type=int, default=40, help="protocol #tasks for tinyImageNet dataset")
######################################
# Only for PRD baseline
######################################
parser.add_argument("--supcon_temperature", type=float, default=0.1, help="Temperature for SupConLoss")
parser.add_argument("--hidden_dim", type=int, default=512, help="hidden_dim in Projection")
parser.add_argument("--feat_dim", type=int, default=128, help="out dim of Projection")
parser.add_argument("--distill_temp", type=float, default=1.0, help="temperature for distillation loss")
parser.add_argument("--distill_coef", type=float, default=4.0, help="coefficient for distillation loss")
parser.add_argument("--prototypes_coef", type=float, default=2.0, help="coefficient for prototypes loss")
parser.add_argument("--prototypes_lr", type=float, default=None, help="lr for prototypes")
parser.add_argument("--num_layers", type=int, default=3, help="#layers in projection")
parser.add_argument("--warmup_epochs", type=int, default=None, help="warmup epochs on first task")

# Read Setting
args = parser.parse_args(sys.argv[1:])
args.increment = 200 // args.num_tasks

# Create folders for storage
if args.source_dir == " ":
    os.chdir(os.getcwd())
else:
    os.chdir(args.source_dir)

mkdir_if_missing(args.log_dir)
mkdir_if_missing(args.tab_dir)
mkdir_if_missing(args.tune_dir)
args.log_dir = os.path.join(args.log_dir, args.folderName)
args.tab_dir = os.path.join(args.tab_dir, args.folderName)
args.tune_dir = os.path.join(args.tune_dir, args.folderName)
mkdir_if_missing(args.log_dir)
mkdir_if_missing(args.tab_dir)
mkdir_if_missing(args.tune_dir)

n_task = args.num_tasks
reg_coef_list = args.reg_coef
l1_hyp_list = args.l1_hyp
if args.agent_name == "MTL":
    args.schedule_gamma = 0.95 ** (
        args.batch_size / (50000 * args.train_size * (1 - args.valid_size))
    )
else:
    args.schedule_gamma = 0.95 ** (
        args.batch_size / (50000 / n_task * args.train_size * (1 - args.valid_size))
    )

n_task = round(n_task * args.stoptask_prop)

# Reproduce experiment
np.random.seed(args.seed)
seed_list = np.random.randint(1, 10000, size=(args.repeat))
nn_seed_list = np.random.randint(1, 10000, size=(args.repeat))

# Start and record
full_val_acc = {}
full_test_acc = {}
average_val_acc = {}
average_test_acc = {}
rank_history = {}
increase_size_history = {}

dt_string = datetime.now().strftime("%m%d_%H%M")
if args.fixed_order:
    file_prefix = (
        "fixed_order_" + args.agent_name + "_" + args.method_desc + "_" + dt_string
    )
else:
    file_prefix = (
        "Nofixed_order_seed_{}_".format(args.seed)
        + args.agent_name
        + "_"
        + args.method_desc
        + "_"
        + dt_string
    )

with Tee(os.path.join(args.log_dir, file_prefix + ".log")):
    print("Split tinyImageNet Experiment information for the method:", args.agent_name)
    print(
        "SEED:",
        args.seed,
        "\nREPEAT:",
        args.repeat,
        "\nTrain subset proportion:",
        args.train_size,
        "\nValid split proportion:",
        args.valid_size,
        "\nEpoch:",
        args.n_epochs,
        "\nBatchSize:",
        args.batch_size,
        "\nlr:",
        args.lr,
        "\ngamma:{:.4f}".format(args.schedule_gamma),
    )
    print("l1_hyp_list:", l1_hyp_list)
    print("prune method:", args.prune_method, "threshold:", args.prune_value)
    print("reg_coef_list:", reg_coef_list)

    total_runtime = Timer()
    total_runtime.tic()

    for l1_hyp in l1_hyp_list:
        args.l1_hyp = l1_hyp
        full_val_acc[l1_hyp] = {}
        full_test_acc[l1_hyp] = {}
        average_val_acc[l1_hyp] = {}
        average_test_acc[l1_hyp] = {}
        rank_history[l1_hyp] = {}
        increase_size_history[l1_hyp] = {}

        for reg_coef in reg_coef_list:
            print(
                "\n========= Start Reg_coef:", reg_coef, "l1_hyp:", l1_hyp, "==========="
            )
            args.reg_coef = reg_coef
            full_val_acc[l1_hyp][reg_coef] = np.zeros((args.repeat, n_task, n_task))
            full_test_acc[l1_hyp][reg_coef] = np.zeros((args.repeat, n_task, n_task))
            average_val_acc[l1_hyp][reg_coef] = np.zeros(args.repeat)
            average_test_acc[l1_hyp][reg_coef] = np.zeros(args.repeat)
            rank_history[l1_hyp][reg_coef] = np.zeros((args.repeat, n_task, 200))
            increase_size_history[l1_hyp][reg_coef] = np.zeros((args.repeat, n_task))

            for r in range(args.repeat):
                print(
                    "\n====== Prepare experiment repeats:",
                    r + 1,
                    "/",
                    args.repeat,
                    "======",
                )
                # # Prepare dataloaders
                (
                    loader,
                    train_loader_splits,
                    valid_loader_splits,
                    test_loader_splits,
                    task_output_space,
                    tasks_info,
                ) = get_loader(
                    args.data_dir,
                    seed=args.seed,
                    fixed_order=False,
                    pc_valid=args.valid_size,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=False if args.gpuid[0] < 0 else True,
                    loader_type="class_incremental_loader",
                    increment=args.increment,
                )

                agent_config = {
                    "n_epochs": args.n_epochs,
                    "lr": args.lr,
                    "schedule_gamma": args.schedule_gamma,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "optimizer": args.optimizer,
                    "out_dim": task_output_space,
                    "print_freq": args.print_freq,
                    "epoch_freq": args.epoch_freq,
                    "gpuid": args.gpuid,
                    "rankMethod": args.rankMethod,
                    "upper_rank": args.upper_rank,
                    "estRank_epoch": args.estRank_epoch,
                    "approxiRate": args.approxiRate,
                    "l1_hyp": args.l1_hyp,
                    "reg_coef": args.reg_coef,
                    "wd_rate": args.wd_rate,
                    "prune_method": args.prune_method,
                    "prune_boundAlltask": args.prune_boundAlltask,
                    "prune_value": args.prune_value,
                    "tune_dir": args.tune_dir,
                    "check_lepoch": args.check_lepoch,
                    "model": args.model,
                    "sparsity": args.sparsity,  # for WSN
                    "img_sz": 64,  # 64 for tinyImageNet
                    'supcon_temperature': args.supcon_temperature, # for PRD baseline
                    'hidden_dim': args.hidden_dim, # for PRD baseline
                    'feat_dim': args.feat_dim, # for PRD baseline
                    'distill_temp': args.distill_temp, # for PRD baseline
                    'distill_coef': args.distill_coef, # for PRD baseline
                    'prototypes_coef': args.prototypes_coef, # for PRD baseline
                    'prototypes_lr': args.prototypes_lr, # for PRD baseline
                    'num_layers': args.num_layers, # for PRD baseline
                    'warmup_epochs': args.warmup_epochs, # for PRD baseline
                }
                torch.manual_seed(nn_seed_list[r])
                agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](
                    agent_config
                )

                val_timer = Timer()
                # Learn MTL
                if args.agent_name == "MTL":
                    agent.learn_batch(train_loader_splits, valid_loader_splits)
                    increase_size_history[l1_hyp][reg_coef][
                        r, :
                    ] = agent.increase_size_rate
                    print(
                        "===== Val_ACC (1st Row) and Test_ACC (2nd Row) on trained tasks ====="
                    )
                    val_timer.tic()
                    for eval_id in range(n_task):
                        full_val_acc[l1_hyp][reg_coef][r, :, eval_id] = agent.validation(
                            valid_loader_splits[eval_id], eval_id
                        )["acc"]
                        full_test_acc[l1_hyp][reg_coef][r, :, eval_id] = agent.validation(
                            test_loader_splits[eval_id], eval_id
                        )["acc"]

                    task_id = n_task - 1
                    print("Task: 0 to", task_id)
                    print(
                        np.stack(
                            (
                                full_val_acc[l1_hyp][reg_coef][
                                    r, task_id, : (task_id + 1)
                                ],
                                full_test_acc[l1_hyp][reg_coef][
                                    r, task_id, : (task_id + 1)
                                ],
                            )
                        ).round(2)
                    )
                    if task_id > 0:
                        tab_forget = (
                            np.diag(
                                full_test_acc[l1_hyp][reg_coef][
                                    r, :task_id, :task_id
                                ].reshape(task_id, task_id)
                            )
                            - full_test_acc[l1_hyp][reg_coef][r, task_id, :task_id]
                        )
                        print("Forgetting:", tab_forget.round(2))
                    print("Evaluation done in {time:.2f}s".format(time=val_timer.toc()))
                else:
                    for task_id in range(n_task):
                        print(
                            "\n===== Task:",
                            task_id,
                            "-- Reg_coef:",
                            reg_coef,
                            "-- l1_hyp:",
                            l1_hyp,
                            "===== Repeat:",
                            r,
                        )

                        # Learn CL
                        if task_id > 0:
                            agent.learn_batch(
                                train_loader_splits[task_id],
                                task_id,
                                valid_loader_splits[task_id],
                                valid_loader_splits[task_id - 1],
                                task_id - 1,
                            )

                            increase_size_history[l1_hyp][reg_coef][
                                r, task_id
                            ] = agent.increase_size_rate
                            if args.agent_name in ["FHALRP", "FLRP"]:
                                for i_rank, (_, rank_l) in enumerate(
                                    agent.rank_current.items()
                                ):
                                    rank_history[l1_hyp][reg_coef][
                                        r, task_id, i_rank
                                    ] = rank_l.item()
                            elif args.agent_name in ["BHALRP", "BLRP"]:
                                for i_rank, (_, rank_l) in enumerate(
                                    agent.rank_previous.items()
                                ):
                                    rank_history[l1_hyp][reg_coef][
                                        r, task_id, i_rank
                                    ] = rank_l.item()
                        else:
                            agent.learn_batch(
                                train_loader_splits[task_id],
                                task_id,
                                valid_loader_splits[task_id],
                            )

                        print(
                            "===== Val_ACC (1st Row) and Test_ACC (2nd Row) on trained tasks ====="
                        )
                        val_timer.tic()
                        # test on all previsous tasks
                        for eval_id in range(task_id + 1):
                            if "WSN" in args.agent_name:
                                full_val_acc[l1_hyp][reg_coef][
                                    r, task_id, eval_id
                                ] = agent.validation(
                                    valid_loader_splits[eval_id],
                                    eval_id,
                                    curr_task_masks=agent.per_task_masks[eval_id],
                                    mode="test",
                                )[
                                    "acc"
                                ]
                                full_test_acc[l1_hyp][reg_coef][
                                    r, task_id, eval_id
                                ] = agent.validation(
                                    test_loader_splits[eval_id],
                                    eval_id,
                                    curr_task_masks=agent.per_task_masks[eval_id],
                                    mode="test",
                                )[
                                    "acc"
                                ]
                            else:
                                full_val_acc[l1_hyp][reg_coef][
                                    r, task_id, eval_id
                                ] = agent.validation(
                                    valid_loader_splits[eval_id], eval_id
                                )[
                                    "acc"
                                ]
                                full_test_acc[l1_hyp][reg_coef][
                                    r, task_id, eval_id
                                ] = agent.validation(
                                    test_loader_splits[eval_id], eval_id
                                )[
                                    "acc"
                                ]

                        print("Task: 0 to", task_id)
                        for row in range(task_id + 1):
                            print("Acc(Test) =\t", end="")
                            for col in range(row + 1):
                                print(
                                    "{:5.2f} ".format(
                                        full_test_acc[l1_hyp][reg_coef][r, row, col]
                                    ),
                                    end="",
                                )
                            print()
                        print(
                            np.stack(
                                (
                                    full_val_acc[l1_hyp][reg_coef][
                                        r, task_id, : (task_id + 1)
                                    ],
                                    full_test_acc[l1_hyp][reg_coef][
                                        r, task_id, : (task_id + 1)
                                    ],
                                )
                            ).round(2)
                        )
                        if task_id > 0:
                            tab_forget = (
                                np.diag(
                                    full_test_acc[l1_hyp][reg_coef][
                                        r, :task_id, :task_id
                                    ].reshape(task_id, task_id)
                                )
                                - full_test_acc[l1_hyp][reg_coef][r, task_id, :task_id]
                            )
                            print("Forgetting:", tab_forget.round(2))
                        print(
                            "Evaluation done in {time:.2f}s".format(time=val_timer.toc())
                        )

                # Calculate average performance across tasks
                average_val_acc[l1_hyp][reg_coef][r] = full_val_acc[l1_hyp][reg_coef][
                    r, task_id, :
                ].mean()
                average_test_acc[l1_hyp][reg_coef][r] = full_test_acc[l1_hyp][reg_coef][
                    r, task_id, :
                ].mean()

                # Print the summary for all repeats
                print(
                    "====== Summary of experiment repeats:",
                    r + 1,
                    "/",
                    args.repeat,
                    "======",
                )
                print("Reg approximate coef:", args.reg_coef, " -- l1_hyp:", l1_hyp)
                for task_id in range(n_task):
                    print(
                        "Task {0} one full train average "
                        "val/test accuracy : {1:.2f}, {2:.2f}".format(
                            task_id,
                            full_val_acc[l1_hyp][reg_coef][r, task_id:, task_id].mean(),
                            full_test_acc[l1_hyp][reg_coef][r, task_id:, task_id].mean(),
                        )
                    )
                print(
                    "Average valid accuracy at final task:",
                    np2word(average_val_acc[l1_hyp][reg_coef]),
                )
                if (r + 1) == args.repeat:
                    print(
                        "mean: {:.2f},".format(average_val_acc[l1_hyp][reg_coef].mean()),
                        "std: {:.2f}".format(average_val_acc[l1_hyp][reg_coef].std()),
                    )

                print(
                    "Average test accuracy at final task:",
                    np2word(average_test_acc[l1_hyp][reg_coef]),
                )
                if (r + 1) == args.repeat:
                    print(
                        "mean: {:.2f},".format(average_test_acc[l1_hyp][reg_coef].mean()),
                        "std: {:.2f}".format(average_test_acc[l1_hyp][reg_coef].std()),
                    )
                tab_forget = (
                    np.diag(
                        full_test_acc[l1_hyp][reg_coef][r, :, :].reshape(n_task, n_task)
                    )
                    - full_test_acc[l1_hyp][reg_coef][r, n_task - 1, :n_task]
                )
                print("Forgetting:", tab_forget.round(2))
                print("Size:", increase_size_history[l1_hyp][reg_coef][r, :])

    # Summary for all regularization parameters
    print("\n============ Details of each run ============")
    for r in range(args.repeat):
        for l1_hyp in l1_hyp_list:
            for reg_coef in reg_coef_list:
                tab_forget = (
                    np.diag(
                        full_test_acc[l1_hyp][reg_coef][r, :, :].reshape(n_task, n_task)
                    )
                    - full_test_acc[l1_hyp][reg_coef][r, n_task - 1, :n_task]
                )
                print(
                    "r=",
                    r,
                    "-- l1_hyp: {},".format(l1_hyp),
                    "-- reg_coef: {},".format(reg_coef),
                    "-- mean: {}".format(average_test_acc[l1_hyp][reg_coef][r].round(2)),
                    "\ntask: {},".format(
                        full_test_acc[l1_hyp][reg_coef][r, n_task - 1, :n_task].round(2)
                    ),
                    "\nF: {}".format(tab_forget.round(2)),
                    "\nSize:",
                    increase_size_history[l1_hyp][reg_coef][r, :],
                    "\n",
                )

    print("\n====== Summary of Regularization Parameters ======")
    for l1_hyp in l1_hyp_list:
        print("\nl1_hyp:", l1_hyp)
        for reg_coef in reg_coef_list:
            tab_forget = 0.0
            for r in range(args.repeat):
                tab_forget += (
                    np.diag(
                        full_test_acc[l1_hyp][reg_coef][r, :, :].reshape(n_task, n_task)
                    )
                    - full_test_acc[l1_hyp][reg_coef][r, n_task - 1, :n_task]
                )
            tab_forget = tab_forget / (args.repeat + 1)
            print(
                'l1_hyp: {},'.format(l1_hyp),
                "reg_coef: {},".format(reg_coef),
                "Tmean: {:.2f},".format(average_test_acc[l1_hyp][reg_coef].mean()),
                "Tstd: {:.2f}".format(average_test_acc[l1_hyp][reg_coef].std()),
                "Vmean: {:.2f},".format(average_val_acc[l1_hyp][reg_coef].mean()),
                "Vstd: {:.2f}".format(average_val_acc[l1_hyp][reg_coef].std()),
                "Size: {:.2f}".format(
                    increase_size_history[l1_hyp][reg_coef][:, n_task - 1].mean()
                ),
                "F: {:.2f}".format(tab_forget.mean())
            )

    # Save acccuracy full table
    with open(
        os.path.join(args.tab_dir, file_prefix + "_full_val_acc_" + ".dict"), "wb"
    ) as table_file:
        pickle.dump(full_val_acc, table_file)

    with open(
        os.path.join(args.tab_dir, file_prefix + "_full_test_acc_" + ".dict"), "wb"
    ) as table_file:
        pickle.dump(full_test_acc, table_file)

    if args.agent_name in ["HALRP"]:
        with open(
            os.path.join(args.tab_dir, file_prefix + "_rank_" + ".dict"), "wb"
        ) as table_file:
            pickle.dump(rank_history, table_file)

    with open(
        os.path.join(args.tab_dir, file_prefix + "_increase_size_" + ".dict"), "wb"
    ) as table_file:
        pickle.dump(increase_size_history, table_file)

    print("* Total time {time:.2f}".format(time=total_runtime.toc()))
    gc.collect()

    print("\n" * 30)
