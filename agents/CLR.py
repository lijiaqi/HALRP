# -*- coding: utf-8 -*-
""" 
    This file is mainly based on the repo: 
        https://github.com/gyhandy/Channel-wise-Lightweight-Reprogramming
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy

from utils.metric import accuracy, AverageMeter, Timer
from models.CLR_models.clr import set_BN
from models.CLR_models.lenet import LeNet, WideLeNet, CLR_LeNet
from models.CLR_models.alexnet import AlexNet, CLR_AlexNet
from models.CLR_models.resnet import ResNet18_flat, CLR_ResNet18


class CLR(nn.Module):

    def __init__(self, agent_config):
        super(CLR, self).__init__()
        self.log = (
            print if agent_config["print_freq"] > 0 else lambda *args: None
        )  # Use a void function to replace the print
        self.config = agent_config
        self.config["n_epochs"] = self.config["n_epochs"]
        self.wd_rate = torch.tensor(agent_config["wd_rate"])
        self.schedule_gamma = agent_config["schedule_gamma"]
        self.num_tasks = len(self.config["out_dim"])
        self.cpt = list(self.config["out_dim"].values())

        self.trained_task = -1
        self.model = self.create_model()
        print(self.model)

        self.model_alltask = nn.ModuleDict()
        self.head_alltask = nn.ModuleDict()

        # Initialize for all tasks
        self.baseSize = sum(p.numel() for p in self.model.parameters())
        self.increase_size_rate = 0.0

        if agent_config["gpuid"][0] >= 0:
            self.cuda()
            self.gpu = True
            self.device = torch.device("cuda")
        else:
            self.gpu = False
            self.device = torch.device("cpu")

        # extra attributes for CLR

    def save_model_task(self, task_id, epoch):
        dir_save = self.config["tune_dir"]
        filename = os.path.join(
            dir_save, "CLR" + "_task" + str(task_id) + "_e" + str(epoch + 1) + ".pth"
        )

        task_state = {"model": self.model_alltask[str(task_id)].state_dict()}
        torch.save(task_state, filename)

    def load_model_task(self, task_id, epoch):
        dir_save = self.config["tune_dir"]
        filename = os.path.join(
            dir_save, "CLR" + "_task" + str(task_id) + "_e" + str(epoch + 1) + ".pth"
        )

        task_state = torch.load(filename)
        self.model_alltask[str(task_id)].load_state_dict(task_state["model"])

        if self.gpu:
            self.cuda()

    def create_model(self):
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        # support for tinyImageNet 64*64
        if "img_sz" in self.config:
            img_sz = self.config["img_sz"]
        else:
            img_sz = 32

        if self.config["model"] == "LeNet":
            model = LeNet(
                num_cls=self.config["out_dim"][0], img_sz=img_sz, track_bn_stats=False
            )
        elif self.config["model"] == "AlexNet":
            model = AlexNet(
                num_cls=self.config["out_dim"][0], img_sz=img_sz, track_bn_stats=False
            )
        elif self.config["model"] == "ResNet18":
            model = ResNet18_flat(
                num_cls=self.config["out_dim"][0],
                nf=32,
                img_sz=img_sz,
                track_bn_stats=False,
            )
        return model

    def build_model_newtask(self, task_id):
        self.log("Building CLR model for task={} ...".format(task_id))
        feat_dim = self.model.backbone_dim
        self.head_alltask[str(task_id)] = nn.Linear(
            feat_dim, self.config["out_dim"][task_id]
        )
        if self.config["model"] == "LeNet":
            CLR_wrapper = CLR_LeNet
        elif self.config["model"] == "AlexNet":
            CLR_wrapper = CLR_AlexNet
        elif self.config["model"] == "ResNet18":
            CLR_wrapper = CLR_ResNet18

        self.model_alltask[str(task_id)] = CLR_wrapper(
            self.model, self.head_alltask[str(task_id)]
        )
        if self.gpu:
            self.model_alltask[str(task_id)] = self.model_alltask[str(task_id)].cuda()

    def init_optimizer(self, task_id, is_pretrain=False):
        if is_pretrain:
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = [
                p
                for p in self.model_alltask[str(task_id)].parameters()
                if p.requires_grad
            ]
        optimizer_arg = {
            "params": params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }

        if self.config["optimizer"] in ["SGD", "RMSprop"]:
            optimizer_arg["momentum"] = self.config["momentum"]
        elif self.config["optimizer"] in ["Rprop"]:
            optimizer_arg.pop("weight_decay")
        elif self.config["optimizer"] == "amsgrad":
            optimizer_arg["amsgrad"] = True
            self.config["optimizer"] = "Adam"

        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.schedule_gamma
        )

    def update_model(self, inputs, targets, task_id, regularization=True):
        loss, outputs = self.criterion(
            inputs=inputs,
            targets=targets,
            task_id=task_id,
            regularization=regularization,
        )
        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param_groups in self.optimizer.param_groups:
                for param in param_groups["params"]:
                    if param.grad is not None:
                        param.grad *= 10.0
        self.optimizer.step()
        self.scheduler.step()  # pytorch 1.5 revised
        return loss.detach(), outputs

    def criterion(self, inputs, targets, task_id, regularization=True):
        outputs = self.model_alltask[str(task_id)].forward(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if regularization:
            reg_wd = torch.tensor(0.0).cuda() if self.gpu else torch.tensor(0.0)
            for name, module in self.model_alltask[str(task_id)].features.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    reg_wd += (module.weight**2).sum() / 2.0
            loss += self.wd_rate * reg_wd
        return loss, outputs

    def learn_first_task(self, train_loader, task_id, val_loader=None, regularization=True):
        self.log("Optimizer is reset for pre-training !")
        self.init_optimizer(task_id=0, is_pretrain=True)

        self.log("Epoch\tBatch\tLoss\ttraAcc\tvalAcc\ttraT\tdataT\tvalT")
        print_freq = self.config["print_freq"]
        epoch_freq = self.config["epoch_freq"]
        acc_max = 0.0
        loss_min = 1e5
        best_idx = 0
        for epoch in range(self.config["n_epochs"]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.model.train()

            # Learning with mini-batch
            data_timer.tic()
            for i, (inputs, targets) in enumerate(train_loader):
                bsz = inputs.shape[0]
                batch_timer.tic()
                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                # loss, outputs = self.update_model(inputs, targets, task_id)
                ### what we do in self.update_model()
                outputs = self.model.forward(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                if regularization:
                    reg_wd = torch.tensor(0.0).cuda() if self.gpu else torch.tensor(0.0)
                    for name, module in self.model.features.named_modules():
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            reg_wd += (module.weight**2).sum() / 2.0
                    loss += self.wd_rate * reg_wd
                self.optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param_groups in self.optimizer.param_groups:
                        for param in param_groups["params"]:
                            if param.grad is not None:
                                param.grad *= 10.0
                self.optimizer.step()
                self.scheduler.step()
                ###
                inputs = inputs.detach()
                targets = targets.detach()

                # measure accuracy and record loss
                acc.update(accuracy(outputs, targets), len(targets))
                losses.update(loss, inputs.size(0))
                # self.scheduler.step() #pytorch 1.5 revised

                batch_time.update(batch_timer.toc())  # measure elapsed time

                if (epoch == 0 and i == 0) or (
                    (epoch % epoch_freq) == 0
                    and print_freq > 0
                    and (i % print_freq) == (print_freq - 1)
                    and i != (len(train_loader) - 1)
                ):
                    self.log(
                        "[{0}/{1}]\t"
                        "[{2}/{3}]\t"
                        "{loss:.2f}\t"
                        "{acc:.2f}\t".format(
                            epoch + 1,
                            self.config["n_epochs"],
                            i + 1,
                            len(train_loader),
                            loss=losses.val,
                            acc=acc.val,
                        )
                    )
                data_timer.tic()

            # Evaluate the performance of current task
            if (epoch % epoch_freq) == 0 or ((epoch + 1) == self.config["n_epochs"]):
                if val_loader != None:
                    val_result = self.validation(val_loader, task_id, is_pretrain=True)
                    val_acc = val_result["acc"]
                    val_time = val_result["time"]

                    # Save best model
                    if (acc_max < val_acc) and (
                        epoch >= (self.config["n_epochs"] - self.config["check_lepoch"])
                    ):
                        acc_max = val_acc
                        best_idx = epoch
                        self.save_model_task(task_id, epoch)
                    else:
                        pass

                else:
                    val_acc, val_time = (0, 0)

                self.log(
                    "[{0}/{1}]\t"
                    "[{2}/{3}]\t"
                    "{loss:.2f}\t"
                    "{acc:.2f}\t"
                    "{val_acc:.2f}\t"
                    "{train_time:.1f}s\t"
                    "{data_time:.1f}s\t"
                    "{val_time:.1f}s".format(
                        epoch + 1,
                        self.config["n_epochs"],
                        i + 1,
                        len(train_loader),
                        loss=losses.val,
                        acc=acc.val,
                        val_acc=val_acc,
                        train_time=batch_time.sum,
                        data_time=data_time.sum,
                        val_time=val_time,
                    )
                )

        # after pretraining, freeze parameters except for BN layers
        for p in self.model.parameters():
            p.requires_grad = False
        set_BN(self.model, requires_grad=True)

    def learn_batch(
        self, train_loader, task_id, val_loader=None, preval_loader=None, preval_id=None
    ):

        self.trained_task += 1
        if self.trained_task == 0:
            self.learn_first_task(train_loader, task_id, val_loader)
        self.build_model_newtask(task_id)
        self.log("Optimizer is reset for the new task!")
        self.init_optimizer(task_id, is_pretrain=False)  # Reset optimizer

        self.log("Epoch\tBatch\tLoss\ttraAcc\tvalAcc\ttraT\tdataT\tvalT")
        print_freq = self.config["print_freq"]
        epoch_freq = self.config["epoch_freq"]
        acc_max = 0.0
        loss_min = 1e5
        best_idx = 0
        for epoch in range(self.config["n_epochs"]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.model_alltask[str(task_id)].train()

            # Learning with mini-batch
            data_timer.tic()
            for i, (inputs, targets) in enumerate(train_loader):
                bsz = inputs.shape[0]
                batch_timer.tic()
                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                loss, outputs = self.update_model(inputs, targets, task_id)

                inputs = inputs.detach()
                targets = targets.detach()

                # measure accuracy and record loss
                acc.update(accuracy(outputs, targets), len(targets))
                losses.update(loss, inputs.size(0))
                # self.scheduler.step() #pytorch 1.5 revised

                batch_time.update(batch_timer.toc())  # measure elapsed time

                if (epoch == 0 and i == 0) or (
                    (epoch % epoch_freq) == 0
                    and print_freq > 0
                    and (i % print_freq) == (print_freq - 1)
                    and i != (len(train_loader) - 1)
                ):
                    self.log(
                        "[{0}/{1}]\t"
                        "[{2}/{3}]\t"
                        "{loss:.2f}\t"
                        "{acc:.2f}\t".format(
                            epoch + 1,
                            self.config["n_epochs"],
                            i + 1,
                            len(train_loader),
                            loss=losses.val,
                            acc=acc.val,
                        )
                    )
                data_timer.tic()

            # Evaluate the performance of current task
            if (epoch % epoch_freq) == 0 or ((epoch + 1) == self.config["n_epochs"]):
                if val_loader != None:
                    val_result = self.validation(val_loader, task_id, is_pretrain=False)
                    val_acc = val_result["acc"]
                    val_time = val_result["time"]

                    # Save best model
                    if (acc_max < val_acc) and (
                        epoch >= (self.config["n_epochs"] - self.config["check_lepoch"])
                    ):
                        acc_max = val_acc
                        best_idx = epoch
                        self.save_model_task(task_id, epoch)
                    else:
                        pass

                else:
                    val_acc, val_time = (0, 0)

                self.log(
                    "[{0}/{1}]\t"
                    "[{2}/{3}]\t"
                    "{loss:.2f}\t"
                    "{acc:.2f}\t"
                    "{val_acc:.2f}\t"
                    "{train_time:.1f}s\t"
                    "{data_time:.1f}s\t"
                    "{val_time:.1f}s".format(
                        epoch + 1,
                        self.config["n_epochs"],
                        i + 1,
                        len(train_loader),
                        loss=losses.val,
                        acc=acc.val,
                        val_acc=val_acc,
                        train_time=batch_time.sum,
                        data_time=data_time.sum,
                        val_time=val_time,
                    )
                )

        # Retrieve the best model
        if (val_loader != None) and (self.config["check_lepoch"] > 0):
            self.log(
                "Retrieve: the best model is at epoch:",
                best_idx + 1,
                "with acc:",
                acc_max,
                "loss:",
                loss_min,
            )
            self.load_model_task(task_id, best_idx)
        self.increase_size_rate = round(
            self.count_parameter(task_id) / self.baseSize - 1, 3
        )
        self.log("Increased parameters so far over base model:", self.increase_size_rate)

    def count_parameter(self, task_id):
        totalNum_param = 0
        totalNum_param += sum(p.numel() for p in self.model_alltask.parameters())
        # for t_id in range(task_id+1):
        # totalNum_param += sum(p.numel() for p in self.head_alltask[str(t_id)].parameters())

        return totalNum_param

    def validation(self, dataloader, task_id, is_pretrain=False):
        assert task_id <= self.trained_task, "task {} has not been trained yet".format(
            task_id
        )
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(dataloader):
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                ### TODO: add for CLR
                if is_pretrain:
                    logits = self.model.forward(inputs)
                else:
                    logits = self.model_alltask[str(task_id)].forward(inputs)
                acc.update(accuracy(logits, targets), len(targets))
        self.train(orig_mode)

        return {"acc": acc.avg, "time": batch_timer.toc()}

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.wd_rate = self.wd_rate.cuda()
        self.model = self.model.cuda()
        self.model_alltask = self.model_alltask.cuda()
        self.head_alltask = self.head_alltask.cuda()

    def cpu(self):
        self.wd_rate = self.wd_rate.cpu()
        self.model = self.model.cpu()
        self.model_alltask = self.model_alltask.cpu()
        self.head_alltask = self.head_alltask.cpu()
