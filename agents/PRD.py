# -*- coding: utf-8 -*-
""" 
    This file is mainly based on the repo: 
        https://github.com/naderAsadi/Probing-Continual-Learning
"""
from typing import Any, Callable, Dict
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from utils.metric import accuracy, AverageMeter, Timer
import kornia
from models.PRD_models.basic import Prototypes
from models.PRD_models.losses import SupConLoss
from models.PRD_models.alexnet import SupConAlexNet
from models.PRD_models.lenet import SupConLeNet, SupConWideLeNet
from models.PRD_models.resnet import SupConResNet18


class PRD(nn.Module):
    def __init__(self, agent_config):
        super(PRD, self).__init__()
        self.log = (
            print if agent_config["print_freq"] > 0 else lambda *args: None
        )  # Use a void function to replace the print
        self.config = agent_config
        self.wd_rate = torch.tensor(agent_config["wd_rate"])
        self.schedule_gamma = agent_config["schedule_gamma"]
        self.num_tasks = len(self.config["out_dim"])

        if "img_sz" in self.config:
            self.img_sz = self.config["img_sz"]
        else:
            self.img_sz = 32
        if agent_config["gpuid"][0] >= 0:
            self.gpu = True
            self.device = torch.device("cuda")
        else:
            self.gpu = False
            self.device = torch.device("cpu")

        self.model = self.create_model()
        self.train_tf = self.get_train_transforms(self.img_sz)
        self.prototypes = Prototypes(
            feat_dim=self.model.encoder.last_hid,
            n_classes_per_task=self.config["out_dim"][0],
            n_tasks=self.num_tasks,
        )
        print("model (features+projection): ", self.model)
        print("model (prototypes): ", self.prototypes)
        self.baseSize = sum(p.numel() for p in self.model.encoder.parameters())
        self.baseSize += sum(
            p.numel() for p in self.prototypes.heads[str(0)].parameters()
        )
        self.increase_size_rate = 0.0

        self.prev_model = None
        self.prev_prototypes = None
        self.supcon_temperature = self.config["supcon_temperature"]
        self.distill_temp = self.config["distill_temp"]
        self.distill_coef = self.config["distill_coef"]
        self.prototypes_coef = self.config["prototypes_coef"]
        self.prototypes_lr = self.config["prototypes_lr"]

        self.supcon_loss = SupConLoss(
            temperature=self.supcon_temperature, device=self.device
        )
        if self.gpu:
            self.cuda()

        self.trained_task = -1
        self.init_optimizer()

    def create_model(self):
        if self.config["model"] == "AlexNet":
            model = SupConAlexNet(
                head="mlp",
                input_size=self.img_sz,
                feat_dim=self.config["feat_dim"],
                hidden_dim=self.config["hidden_dim"],
                proj_bn=False,
                num_layers=self.config["num_layers"],
            )
        elif self.config["model"] == "ResNet18":
            model = SupConResNet18(
                head="mlp",
                input_size=self.img_sz,
                feat_dim=self.config["feat_dim"],
                hidden_dim=self.config["hidden_dim"],
                proj_bn=False,
                num_layers=self.config["num_layers"],
            )
        elif self.config["model"] == "LeNet":
            model = SupConLeNet(
                head="mlp",
                input_size=self.img_sz,
                in_channel=3,
                feat_dim=self.config["feat_dim"],
                hidden_dim=self.config["hidden_dim"],
                proj_bn=False,
                num_layers=self.config["num_layers"],
            )
        return model

    def get_train_transforms(self, img_sz):
        return torch.nn.Sequential(
            kornia.augmentation.RandomCrop(size=(img_sz, img_sz), padding=4, fill=-1),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.2),
        )

    def init_optimizer(self):
        optimizer_arg = {
            "params": [
                {"params": self.model.parameters()},
                {
                    "params": self.prototypes.parameters(),
                    "lr": self.prototypes_lr,
                    "momentum": 0.0,
                    "weight_decay": 0.0,
                },
            ],
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

    def count_parameter(self, task_id):
        ### encoder + projection
        totalNum_param = sum(p.numel() for p in self.model.parameters())
        for t in range(task_id + 1):
            for param in self.prototypes.heads[str(t)].parameters():
                totalNum_param += param.numel()
        return totalNum_param

    def learn_batch(
        self, train_loader, train_id, val_loader=None, preval_loader=None, preval_id=None
    ):
        self.train()
        self.on_task_start()
        self.trained_task += 1
        self.log("Optimizer is reset for the new task!")
        self.init_optimizer()

        self.log("Epoch\tBatch\tLoss\ttraAcc\tvalAcc\ttraT\tdataT\tvalT")
        print_freq = self.config["print_freq"]
        epoch_freq = self.config["epoch_freq"]
        acc_max = 0.0
        loss_min = 1e5
        best_idx = 0
        n_epochs = self.config["n_epochs"]
        for epoch in range(n_epochs):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            self.train()
            data_timer.tic()
            for i, (inputs, targets) in enumerate(train_loader):
                batch_timer.tic()
                data_time.update(data_timer.toc())

                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                loss, outputs = self.observe(inputs, targets, train_id)

                inputs = inputs.detach()
                targets = targets.detach()
                # measure accuracy and record loss
                acc.update(accuracy(outputs, targets), len(targets))
                losses.update(loss, inputs.size(0))

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
                            n_epochs,
                            i + 1,
                            len(train_loader),
                            loss=losses.val,
                            acc=acc.val,
                        )
                    )
                data_timer.tic()

            # Evaluate the performance of current task
            if (epoch % epoch_freq) == 0 or ((epoch + 1) == self.config["n_epochs"]):
                # if val_loader != None:
                if False:
                    ### TODO: after-epoch validation will decrease final acc. Weird
                    val_result = self.validation(val_loader, train_id)
                    val_acc = val_result["acc"]
                    val_time = val_result["time"]

                    # # Save best model
                    # if (acc_max < val_acc) and (
                    #     epoch >= (self.config["n_epochs"] - self.config["check_lepoch"])
                    # ):
                    #     acc_max = val_acc
                    #     best_idx = epoch
                    #     self.save_model_task(epoch)
                    # else:
                    #     pass

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

        # # Retrieve the best model
        # if (val_loader != None) and (self.config["check_lepoch"] > 0):
        #     self.log(
        #         "Retrieve: the best model is at epoch:",
        #         best_idx + 1,
        #         "with acc:",
        #         acc_max,
        #         "loss:",
        #         loss_min,
        #     )
        #     self.load_model_task(best_idx)

        self.increase_size_rate = round(
            self.count_parameter(train_id) / self.baseSize - 1, 3
        )
        self.log("Increased parameters so far over base model:", self.increase_size_rate)
        self.on_task_finish(train_id)

    def observe(self, inputs, targets, task_id):
        """full step of processing and learning from data"""
        # --- training --- #
        inc_loss, outputs = self.process_inc(inputs, targets, task_id)
        self.optimizer.zero_grad()
        inc_loss.backward()
        with torch.no_grad():
            for param_groups in self.optimizer.param_groups:
                for param in param_groups['params']:
                    if param.grad is not None:
                        param.grad *= 10.0
        self.optimizer.step()
        # self.scheduler.step()
        return inc_loss, outputs.detach()

    def process_inc(self, inputs, targets, task_id) -> torch.FloatTensor:
        """get loss from incoming data"""

        x1, x2 = self.train_tf(inputs), self.train_tf(inputs)
        aug_data = torch.cat((x1, x2), dim=0)  # (2*N, d)
        bsz = inputs.shape[0]

        features = self.model.return_hidden(aug_data)  # feat: (2*N, D)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)

        # SupCon Loss
        proj_features = self.model.forward_classifier(features)  # feat_proj: (2*N, d)
        proj_features = F.normalize(proj_features, dim=1)  # normalize embedding
        proj_f1, proj_f2 = torch.split(proj_features, [bsz, bsz], dim=0)
        proj_features = torch.cat([proj_f1.unsqueeze(1), proj_f2.unsqueeze(1)], dim=1)
        # shape=(N,V,d)
        supcon_loss = self.supcon_loss(proj_features, labels=targets)

        # Distillation loss
        loss_d = self.relation_distillation_loss(
            features, data=aug_data, current_task_id=task_id
        )

        # Prorotypes loss
        loss_p, outputs = self.linear_loss(
            features.detach().clone(),
            labels=targets.repeat(2),
            current_task_id=task_id,
        )
        return (
            supcon_loss + self.prototypes_coef * loss_p + self.distill_coef * loss_d,
            outputs,
        )

    ### loss_d in Eq.(6)
    def _get_scores(
        self, features: torch.FloatTensor, prototypes: Prototypes, task_id: int
    ) -> torch.FloatTensor:

        nobout = F.linear(features, prototypes.heads[str(task_id)].weight)
        wnorm = torch.norm(prototypes.heads[str(task_id)].weight, dim=1, p=2)
        nobout = nobout / wnorm
        return nobout

    def _distillation_loss(
        self, current_out: torch.FloatTensor, prev_out: torch.FloatTensor
    ) -> torch.FloatTensor:

        log_p = torch.log_softmax(current_out / self.distill_temp, dim=1)  # student
        q = torch.softmax(prev_out / self.distill_temp, dim=1)  # teacher
        result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        # result = torch.sum(-q * log_p, dim=-1).mean()

        return result

    ### loss_d in Eq.(6)
    def relation_distillation_loss(
        self, features: torch.FloatTensor, data: torch.FloatTensor, current_task_id: int
    ) -> torch.FloatTensor:
        if self.prev_model is None:
            return 0.0

        old_model_preds = dict()
        new_model_preds = dict()

        with torch.inference_mode():
            old_features = self.prev_model.return_hidden(data)

        for task_id in range(current_task_id):
            with torch.inference_mode():
                old_model_preds[task_id] = self._get_scores(
                    old_features, prototypes=self.prev_prototypes, task_id=task_id
                )
            new_model_preds[task_id] = self._get_scores(
                features, prototypes=self.prototypes, task_id=task_id
            )

        dist_loss = 0
        for task_id in old_model_preds.keys():
            dist_loss += self._distillation_loss(
                current_out=new_model_preds[task_id],
                prev_out=old_model_preds[task_id].clone(),
            )

        return dist_loss

    ### loss_p in Eq.(4)
    def linear_loss(
        self,
        features: torch.FloatTensor,
        labels: torch.Tensor,
        current_task_id: int,
    ) -> torch.FloatTensor:

        # bsz = labels.shape[0] // 2
        # nobout = F.linear(features, self.prototypes.heads[str(current_task_id)].weight) # (2*N, C)
        # wnorm = torch.norm(self.prototypes.heads[str(current_task_id)].weight, dim=1, p=2) # (C, d)->(C,)
        # nobout = nobout / wnorm # (2*N, C)
        # feat_norm = torch.norm(features, dim=1, p=2) # (2*N,)

        # indecies = labels.unsqueeze(1)
        # out = nobout.gather(1, indecies).squeeze() # (2*N, )
        # out = out / feat_norm
        # loss = sum(1 - out) / out.size(0)
        # outputs, _ = torch.split(nobout, [bsz, bsz], dim=0)

        bsz = labels.shape[0] // 2
        nobout = F.linear(
            features, self.prototypes.heads[str(current_task_id)].weight
        )  # (2*N, C)
        wnorm = torch.norm(
            self.prototypes.heads[str(current_task_id)].weight, dim=1, p=2, keepdim=True
        ).T  # (C, d) -> (C, 1) -> (1, C)
        feat_norm = torch.norm(features, dim=1, p=2, keepdim=True)  # (2*N, d) -> (2*N, 1)
        outputs = nobout / wnorm
        outputs = outputs / feat_norm

        indecies = labels.unsqueeze(1)  # (2*N, 1)
        out = outputs.gather(1, indecies).squeeze()
        loss = sum(1 - out) / out.size(0)
        outputs, _ = torch.split(outputs, [bsz, bsz], dim=0)

        return loss, outputs.detach()

    def predict(self, x: torch.FloatTensor, task_id: int = None) -> torch.FloatTensor:
        """used for eval time prediction"""
        features = self.model.return_hidden(x)
        # Copy previous weights
        no_normed_weights = self.prototypes.heads[str(task_id)].weight.data.clone()
        # Normalize weights and features
        self.prototypes.heads[str(task_id)].weight.copy_(
            F.normalize(self.prototypes.heads[str(task_id)].weight.data, dim=1, p=2)
        )
        features = F.normalize(features, dim=1, p=2)  # pass through projection head

        logits = self.prototypes(features, task_id=task_id)
        self.prototypes.heads[str(task_id)].weight.copy_(no_normed_weights)
        return logits, features

    @torch.no_grad()
    def validation(self, eval_loader, eval_id):
        assert eval_id <= self.trained_task, "task {} has not been trained yet".format(
            eval_id
        )
        batch_timer = Timer()
        acc = AverageMeter()
        loss = AverageMeter()
        batch_timer.tic()

        self.eval()
        for i, (data, target) in enumerate(eval_loader):
            data, target = data.to(self.device), target.to(self.device)
            logits, features = self.predict(data, eval_id)

            acc.update(accuracy(logits, target), len(target))
            loss.update(nn.CrossEntropyLoss()(logits, target), len(target))
        self.train()
        return {"acc": acc.avg, "loss": loss.avg, "time": batch_timer.toc()}

    def train(self):
        self.model.train()
        self.prototypes.train()

    def eval(self):
        self.model.eval()
        self.prototypes.eval()

    def on_task_start(self, *args):
        pass

    def on_task_finish(self, task_id: int):
        self.prev_model = copy.deepcopy(self.model)
        self.prev_prototypes = copy.deepcopy(self.prototypes)
        # self.prev_model.eval()
        # self.prev_prototypes.eval()

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.wd_rate = self.wd_rate.cuda()
        self.model = self.model.cuda()
        self.prototypes = self.prototypes.cuda()

    def cpu(self):
        self.wd_rate = self.wd_rate.cpu()
        self.model = self.model.cpu()
        self.prototypes = self.prototypes.cpu()

    def save_model_task(self, epoch):
        dir_save = self.config["tune_dir"]
        filename = os.path.join(dir_save, "PRD" + "_e" + str(epoch + 1) + ".pth")

        task_state = {
            "model": self.model.state_dict(),
            "prototypes": self.prototypes.state_dict(),
        }
        torch.save(task_state, filename)

    def load_model_task(self, epoch):
        dir_save = self.config["tune_dir"]
        filename = os.path.join(dir_save, "PRD" + "_e" + str(epoch + 1) + ".pth")

        task_state = torch.load(filename)
        self.model.load_state_dict(task_state["model"])
        self.prototypes.load_state_dict(task_state["prototypes"])

        if self.gpu:
            self.cuda()
