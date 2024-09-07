import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict 
from utils.metric import accuracy, AverageMeter, Timer
from models.alexnet import AlexNet, rankPerturb4_AlexNet
from models.lenet import LeNet, LargeLeNet, rankPerturb4_LeNet


class HALRP(nn.Module):
    def __init__(self, agent_config):
        super(HALRP, self).__init__()
        self.log = (
            print if agent_config["print_freq"] > 0 else lambda *args: None
        )
        self.config = agent_config
        self.config["n_epochs"] = self.config["n_epochs"]
        self.rankMethod = agent_config["rankMethod"]
        self.approxiRate = agent_config["approxiRate"]
        if agent_config["estRank_epoch"] < 1:
            raise Exception("Not correct estRank_epoch!")
        self.l1_hyp = torch.tensor(agent_config["l1_hyp"])
        self.wd_rate = torch.tensor(agent_config["wd_rate"])
        self.schedule_gamma = agent_config["schedule_gamma"]

        self.model = self.create_model(0)
        self.model_alltask = nn.ModuleDict()
        self.model_alltask[str(0)] = self.model

        self.head_alltask = nn.ModuleDict()
        self.Bias_alltask = {}
        self.BN_alltask = {}
        self.WMask_Sperturb_alltask = {}
        self.WMask_Rperturb_alltask = {}
        self.WBias_Sperturb_alltask = {}
        self.WBias_Rperturb_alltask = {}

        self.rank_current = {}  
        self.weightedLoss_alltask = {}
        self.explainedRate_alltask = {}
        self.param_track = nn.ParameterDict() 
        for i in range(len(self.model.features)):
            if isinstance(
                self.model.features[i],
                (torch.nn.modules.conv.Conv2d, torch.nn.modules.Linear),
            ):
                self.param_track[str(i)] = self.model.features[i].weight

        self.empFI = True
        self.importance_alltask = {} 
        self.baseSize = sum(p.numel() for p in self.model.parameters())
        self.increase_size_rate = 0.0

        self.criterion_fn = nn.CrossEntropyLoss()
        if agent_config["gpuid"][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.trained_task = -1  

    def save_model_task(self, task_id, epoch):
        dir_save = self.config["tune_dir"]
        filename = os.path.join(
            dir_save, "HALRP" + "_task" + str(task_id) + "_e" + str(epoch + 1) + ".pth"
        )
        if task_id == 0:
            task_state = {"model0": self.model_alltask[str(0)].state_dict()}
        else:
            task_state = {
                "head": self.head_alltask[str(task_id)].state_dict(),
                "Bias": self.Bias_alltask[task_id].state_dict(),
                "BN": self.BN_alltask[task_id].state_dict(),
                "WMask_Sperturb_alltask": self.WMask_Sperturb_alltask[task_id].state_dict(),
                "WMask_Rperturb_alltask": self.WMask_Rperturb_alltask[task_id].state_dict(),
                "WBias_Sperturb_alltask": self.WBias_Sperturb_alltask[task_id].state_dict(),
                "WBias_Rperturb_alltask": self.WBias_Rperturb_alltask[task_id].state_dict(),
                "task_id": task_id,
            }
        torch.save(task_state, filename)

    def load_model_task(self, task_id, epoch):
        dir_save = self.config["tune_dir"]
        filename = os.path.join(
            dir_save, "HALRP" + "_task" + str(task_id) + "_e" + str(epoch + 1) + ".pth"
        )
        task_state = torch.load(filename)
        if task_id == 0:
            self.model_alltask[str(0)].load_state_dict(task_state["model0"])
        else:
            self.head_alltask[str(task_id)].load_state_dict(task_state["head"])
            self.BN_alltask[task_id].load_state_dict(task_state["BN"])
            self.Bias_alltask[task_id].load_state_dict(task_state["Bias"])
            self.WMask_Sperturb_alltask[task_id].load_state_dict(
                task_state["WMask_Sperturb_alltask"]
            )
            self.WMask_Rperturb_alltask[task_id].load_state_dict(
                task_state["WMask_Rperturb_alltask"]
            )
            self.WBias_Sperturb_alltask[task_id].load_state_dict(
                task_state["WBias_Sperturb_alltask"]
            )
            self.WBias_Rperturb_alltask[task_id].load_state_dict(
                task_state["WBias_Rperturb_alltask"]
            )
        if self.gpu:
            self.cuda()

    def create_model(self, task_id):
        if "img_sz" in self.config:
            img_sz = self.config["img_sz"]
        else:
            img_sz = 32
        if self.config["model"] == "LeNet":
            model = LeNet(out_dim=self.config["out_dim"][0], img_sz=img_sz)
        elif self.config["model"] == "AlexNet":

            model = AlexNet(out_dim=self.config["out_dim"][task_id], img_sz=img_sz)
        else:
            raise NotImplementedError
        return model

    def update_model_newtask(self, task_id):
        if task_id > 0:
            if self.config["model"] == "LeNet":
                wrapper = rankPerturb4_LeNet
            elif self.config["model"] == "AlexNet":
                wrapper = rankPerturb4_AlexNet
            else:
                raise NotImplementedError
            i = task_id
            self.model_alltask[str(i)] = wrapper(
                self.model_alltask[str(0)],
                self.Bias_alltask[i],
                self.head_alltask[str(i)],
                self.BN_alltask[i],
                self.WMask_Sperturb_alltask[i],
                self.WMask_Rperturb_alltask[i],
                self.WBias_Sperturb_alltask[i],
                self.WBias_Rperturb_alltask[i],
            )
        if self.gpu:
            self.cuda()

    def update_model_alltask(self):
        for i in range(1, self.trained_task + 1):
            if self.config["model"] == "LeNet":
                wrapper = rankPerturb4_LeNet
            elif self.config["model"] == "AlexNet":
                wrapper = rankPerturb4_AlexNet
            else:
                raise NotImplementedError
            self.model_alltask[str(i)] = wrapper(
                self.model_alltask[str(0)],
                self.Bias_alltask[i],
                self.head_alltask[str(i)],
                self.BN_alltask[i],
                self.WMask_Sperturb_alltask[i],
                self.WMask_Rperturb_alltask[i],
                self.WBias_Sperturb_alltask[i],
                self.WBias_Rperturb_alltask[i],
            )
        if self.gpu:
            self.cuda()

    def forward(self, x, task_id):
        assert task_id <= self.trained_task, "task {} has not been trained yet".format(task_id)
        return self.model_alltask[str(task_id)].forward(x)

    def predict(self, inputs, task_id):
        orig_mode = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(inputs, task_id).detach()
        self.train(orig_mode)
        return out

    def new_task(self, task_id):
        self.model = self.create_model(task_id)
        self.model_alltask[str(task_id)] = self.model

        self.head_alltask[str(task_id)] = self.model.last
        Bias = nn.ParameterDict()
        BN = nn.ModuleDict()
        for i in range(len(self.model.features)):
            if isinstance(
                self.model.features[i], (torch.nn.modules.conv.Conv2d, torch.nn.modules.Linear)
            ):
                Bias[str(i)] = self.model.features[i].bias

            if isinstance(
                self.model.features[i], (torch.nn.modules.BatchNorm2d, torch.nn.modules.BatchNorm1d)
            ):
                BN[str(i)] = self.model.features[i]

        self.Bias_alltask[task_id] = Bias
        self.BN_alltask[task_id] = BN

        self.model.features.load_state_dict(self.model_alltask[str(0)].features.state_dict())

        if self.gpu:
            self.cuda()

    def estimate_ranks(self, dataloader, task_id, approxiRate=0.6, val_loader=None):
        self.log("Pretrained for", self.config["estRank_epoch"], "epochs...")
        for epoch in range(self.config["estRank_epoch"]):
            self.model.train()

            for i, (inputs, targets) in enumerate(dataloader):
                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                loss, outputs = self.update_model(inputs, targets, task_id, regularization=False)

        if val_loader is None:
            self.calculate_importance(dataloader)
        else:
            self.calculate_importance(val_loader)

        maskS_dict = {}
        maskR_dict = {}
        for k, p in self.param_track.items():
            wShape = p.shape
            maskR_dict[k] = torch.zeros(wShape[0]).cuda() if self.gpu else torch.zeros(wShape[0])
            maskS_dict[k] = torch.zeros(wShape[1]).cuda() if self.gpu else torch.zeros(wShape[1])
            if isinstance(self.model.features[int(k)], torch.nn.modules.conv.Conv2d):
                w_free = (
                    self.model.features[int(k)].weight.detach().clone()
                )
                w_base = p.detach().clone()
                for j in range(wShape[0]):
                    maskR_dict[k][j] = (w_free[j, :, :, :] *
                                        w_base[j, :, :, :]).sum() / (w_base[j, :, :, :]**2).sum().clamp_(min=1e-20)
                for i in range(wShape[1]):
                    maskS_dict[k][i] = (maskR_dict[k].view(-1, 1, 1, 1) * w_base[:, i, :, :] * w_free[:, i, :, :]).sum() / ((maskR_dict[k].view(-1, 1, 1, 1) * w_base[:, i, :, :])**2).sum().clamp_(min=1e-20)
            elif isinstance(self.model.features[int(k)], torch.nn.modules.Linear):
                w_free = (
                    self.model.features[int(k)].weight.detach().clone()
                )
                w_base = p.detach().clone()
                for j in range(wShape[0]):
                    maskR_dict[k][j] = (w_free[j, :] * w_base[j, :]).sum() / (w_base[j, :]**2).sum().clamp_(min=1e-20)
                for i in range(wShape[1]):
                    maskS_dict[k][i] = (maskR_dict[k].view(-1, 1) * w_base[:, i] * w_free[:, i]).sum() / ((maskR_dict[k].view(-1, 1) * w_base[:, i])**2).sum().clamp_(min=1e-20)

        u_dict = {}
        s_dict = {}
        v_dict = {}
        num_p_layer = torch.zeros(len(self.param_track))
        for i, (k, p) in enumerate(self.param_track.items()):
            wShape = p.shape
            if isinstance(self.model.features[int(k)], torch.nn.modules.conv.Conv2d):
                w_free = (
                    self.model.features[int(k)].weight.detach().clone()
                )
                w_base = p.detach().clone()
                discrepancy = w_free - maskS_dict[k].view(1, -1, 1, 1) * w_base * maskR_dict[
                    k
                ].view(-1, 1, 1, 1)
                discrepancy_reshape = discrepancy.view(
                    wShape[0], wShape[1], wShape[2] * wShape[3]
                ).permute(2, 0, 1)
                mean_discrepancy_reshape = discrepancy_reshape.mean(0)
                u_dict[k], s_dict[k], v_dict[k] = torch.svd(mean_discrepancy_reshape)
            elif isinstance(self.model.features[int(k)], torch.nn.modules.Linear):
                w_free = (
                    self.model.features[int(k)].weight.detach().clone()
                )
                w_base = p.detach().clone()
                discrepancy = w_free - maskS_dict[k].view(1, -1) * w_base * maskR_dict[k].view(
                    -1, 1
                )
                u_dict[k], s_dict[k], v_dict[k] = torch.svd(discrepancy)
            num_p_layer[i] = len(s_dict[k])
        imp_sv = torch.zeros(num_p_layer.sum().type(torch.int))
        idx_start = 0
        idx_end = 0
        scale = 1
        for i, (k, p) in enumerate(self.param_track.items()):
            if isinstance(self.model.features[int(k)], torch.nn.modules.conv.Conv2d):
                scale = p.shape[2] * p.shape[3]
            else:
                scale = 1
            idx_start = idx_start + (0 if i == 0 else num_p_layer[i - 1].int())
            idx_end = idx_end + num_p_layer[i].int()
            imp_sv[idx_start:idx_end] = (
                scale * (s_dict[k] ** 2) * self.importance_alltask[task_id][int(k)]
            )
        # Sort the loss score
        imp_sv_desc, desc_index = imp_sv.sort(descending=True)
        cumRatio_imp_sv_desc = imp_sv_desc.cumsum(0)
        cumRatio_imp_sv_desc = cumRatio_imp_sv_desc / cumRatio_imp_sv_desc[-1]
        truncateR = np.argmax(cumRatio_imp_sv_desc > approxiRate)

        idx_start = 0
        idx_end = 0
        weightedLoss = torch.zeros(len(self.param_track))
        explainedRate = torch.zeros(len(self.param_track))
        cum_p_layer = torch.cat([torch.zeros(1), torch.cumsum(num_p_layer, 0)], 0)
        # Estimate Ranks for achieving required explained variations
        for i, k in enumerate(self.param_track.keys()):
            if self.rankMethod == "flat":
                self.rank_current[int(k)] = torch.tensor(self.config["upper_rank"]).int()
            else:
                self.rank_current[int(k)] = (
                    (
                        (desc_index[: (truncateR + 1)] < cum_p_layer[i + 1])
                        & (desc_index[: (truncateR + 1)] >= cum_p_layer[i])
                    )
                    .sum()
                    .int()
                )
            if self.rank_current[int(k)] == 0:
                self.rank_current[int(k)] = torch.tensor(1)
            elif self.rank_current[int(k)] > self.config["upper_rank"]:
                self.rank_current[int(k)] = torch.tensor(self.config["upper_rank"])
            idx_start = idx_start + (0 if i == 0 else num_p_layer[i - 1].int())
            idx_end = idx_end + num_p_layer[i].int()
            idx_keep = idx_start + self.rank_current[int(k)]
            weightedLoss[i] = imp_sv[idx_start:idx_end].sum()
            explainedRate[i] = imp_sv[idx_start:idx_keep].sum() / weightedLoss[i]

        WMask_Sperturb = nn.ParameterDict()
        WMask_Rperturb = nn.ParameterDict()
        WBias_Sperturb = nn.ParameterDict()
        WBias_Rperturb = nn.ParameterDict()
        for k in self.param_track.keys():
            rank = self.rank_current[int(k)]
            WMask_Sperturb[k] = torch.nn.Parameter(maskS_dict[k], requires_grad=True)
            WMask_Rperturb[k] = torch.nn.Parameter(maskR_dict[k], requires_grad=True)

            WBias_Sperturb[k] = torch.nn.Parameter(
                v_dict[k][:, :rank].transpose(0, 1) * s_dict[k][:rank].sqrt().view(-1, 1),
                requires_grad=True,
            )
            WBias_Rperturb[k] = torch.nn.Parameter(
                u_dict[k][:, :rank] * s_dict[k][:rank].sqrt().view(1, -1), requires_grad=True
            )

        self.WMask_Sperturb_alltask[task_id] = WMask_Sperturb
        self.WMask_Rperturb_alltask[task_id] = WMask_Rperturb
        self.WBias_Sperturb_alltask[task_id] = WBias_Sperturb
        self.WBias_Rperturb_alltask[task_id] = WBias_Rperturb

        if self.gpu:
            self.cuda()

        self.weightedLoss_alltask[task_id] = weightedLoss
        self.explainedRate_alltask[task_id] = explainedRate
        self.log("The weightedLoss for each layer:")
        self.log(weightedLoss)
        self.log("The Low-rank approximation explainedRate for each layer:")
        self.log(explainedRate)
        if self.rankMethod == "flat":
            self.log("Flat ranks selected:", [v.item() for _, v in self.rank_current.items()])
        else:
            self.log(
                "The selected rank for each layer:",
                [v.item() for _, v in self.rank_current.items()],
            )
            self.log("Low rank approximation rate:", approxiRate)
        self.increase_size_rate = round(self.count_parameter() / self.baseSize - 1, 3)
        self.log("Increased parameters so far over base model:", self.increase_size_rate)

    def calculate_importance(self, dataloader):
        self.log("Computing Layerwise Weight Importance...")
        imp_timer = Timer()
        imp_timer.tic()

        importance = OrderedDict()
        for key in self.param_track:
            if self.gpu:
                importance[int(key)] = torch.tensor(0.0).cuda()
            else:
                importance[int(key)] = torch.tensor(0.0)

        mode = self.training
        self.eval()

        for inputs, targets in dataloader:
            if self.gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.model.forward(inputs)
            if self.empFI:
                inds = targets
            else:
                inds = outputs.max(1)[1].flatten()
            loss = self.criterion(outputs, inds, regularization=False)
            self.model.zero_grad()
            loss.backward()

            for i, p in importance.items():
                p += torch.mean(self.model.features[i].weight.grad**2) * len(inputs) / len(dataloader)

        self.train(mode=mode)
        self.importance_alltask[self.trained_task] = importance

        self.log(torch.tensor([v.item() for k, v in importance.items()]))
        self.log("Done in {time:.2f}s".format(time=imp_timer.toc()))

    def criterion(self, preds, targets, regularization=True):
        loss = self.criterion_fn(preds, targets)

        if regularization:
            reg_wd = torch.tensor(0.0).cuda() if self.gpu else torch.tensor(0.0)
            reg_sparse = torch.tensor(0.0).cuda() if self.gpu else torch.tensor(0.0)
            if self.trained_task == 0:
                for i in range(len(self.model.features)):
                    if isinstance(
                        self.model.features[i],
                        (torch.nn.modules.conv.Conv2d, torch.nn.modules.Linear),
                    ):
                        reg_wd += (self.model.features[i].weight ** 2).sum() / 2.0

            elif self.trained_task > 0:
                for _, param in self.WMask_Sperturb_alltask[self.trained_task].items():
                    reg_wd += (param**2).sum() / 2.0
                for _, param in self.WMask_Rperturb_alltask[self.trained_task].items():
                    reg_wd += (param**2).sum() / 2.0

                for _, param in self.WBias_Sperturb_alltask[self.trained_task].items():
                    reg_wd += (param**2).sum() / 2.0
                    reg_sparse += (torch.abs(param)).sum()
                for _, param in self.WBias_Rperturb_alltask[self.trained_task].items():
                    reg_wd += (param**2).sum() / 2.0
                    reg_sparse += (torch.abs(param)).sum()
            loss += self.l1_hyp * reg_sparse + self.wd_rate * reg_wd
        return loss

    def init_optimizer(self, lowRank=False):
        if lowRank:
            task_id = self.trained_task
            pertubation_param = []
            for param in self.head_alltask[str(task_id)].parameters():
                pertubation_param.append(param)
            for _, param in self.Bias_alltask[task_id].items():
                pertubation_param.append(param)
            for _, bnLayer in self.BN_alltask[task_id].items():
                for param in bnLayer.parameters():
                    pertubation_param.append(param)
            for _, param in self.WMask_Sperturb_alltask[task_id].items():
                pertubation_param.append(param)
            for _, param in self.WMask_Rperturb_alltask[task_id].items():
                pertubation_param.append(param)
            for _, param in self.WBias_Sperturb_alltask[task_id].items():
                pertubation_param.append(param)
            for _, param in self.WBias_Rperturb_alltask[task_id].items():
                pertubation_param.append(param)
            optimizer_arg = {
                "params": pertubation_param,
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            }
        else:
            optimizer_arg = {
                "params": filter(lambda p: p.requires_grad, self.model.parameters()),
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

    def l1_pruning(self, pv, pmethod):
        self.log(
            "Pruning method:", pmethod, "--Bound all tasks:", self.config["prune_boundAlltask"]
        )
        with torch.no_grad():
            if pmethod == "absolute":
                value = pv[0]
                self.log("Pruning task-adaptive param bias with value <=", value, "...")
            elif pmethod in ["relativeAllLayerAllTask", "relativeAllLayer", "relative"]:
                if self.config["prune_boundAlltask"]:
                    increase_size = self.count_parameter(sparse=False) / self.baseSize - 1
                    if increase_size < (1 - pv[0]):
                        self.log("No pruning is needed as overall increament <=", 1 - pv[0])
                        return None
                    else:
                        rate = 1 - (1 - pv[0]) / increase_size
                else:
                    rate = pv[0]

                if pmethod == "relative":
                    self.log(
                        "Pruning task-adaptive param bias with value <=",
                        round(rate, 3),
                        "-quantile of each layer...",
                    )
                elif pmethod == "relativeAllLayer":
                    self.log(
                        "Pruning task-adaptive param bias with value <=",
                        round(rate, 3),
                        "-quantile of all layers...",
                    )
                elif pmethod == "relativeAllLayerAllTask":
                    self.log(
                        "Pruning task-adaptive param bias with value <=",
                        round(rate, 3),
                        "-quantile of all layers of all tasks...",
                    )
            elif pmethod in ["mixAR_AllLayerAllTask", "mixAR_AllLayer", "mixAR"]:
                value = pv[0]
                if self.config["prune_boundAlltask"]:
                    increase_size = self.count_parameter(sparse=False) / self.baseSize - 1
                    if increase_size < (1 - pv[1]):
                        self.log("No pruning is needed as overall increament <=", 1 - pv[1])
                        return None
                    else:
                        rate = 1 - (1 - pv[1]) / increase_size
                else:
                    rate = pv[1]
                if pmethod == "mixAR":
                    self.log(
                        "Pruning task-adaptive param bias with value <=",
                        value,
                        "; and",
                        round(rate, 3),
                        "-quantile of each layer...",
                    )
                elif pmethod == "mixAR_AllLayer":
                    self.log(
                        "Pruning task-adaptive param bias with value <=",
                        value,
                        "; and",
                        round(rate, 3),
                        "-quantile of all layers...",
                    )
                elif pmethod == "mixAR_AllLayerAllTask":
                    self.log(
                        "Pruning task-adaptive param bias with value <=",
                        value,
                        "; and",
                        round(rate, 3),
                        "-quantile of all layers of all tasks...",
                    )
            else:
                raise Exception("Not correct pruning method!")

            if pmethod in ["relativeAllLayerAllTask", "mixAR_AllLayerAllTask"]:
                param_AllLayerAllTask = ()
                for task_id in range(1, self.trained_task + 1):
                    for key in self.WBias_Rperturb_alltask[task_id].keys():
                        paramS = self.WBias_Sperturb_alltask[task_id][key]
                        paramR = self.WBias_Rperturb_alltask[task_id][key]
                        param_AllLayerAllTask = param_AllLayerAllTask + (
                            paramS.data.reshape(-1),
                            paramR.data.reshape(-1),
                        )
                if pmethod == "relativeAllLayerAllTask":
                    threshold = np.percentile(
                        torch.cat(param_AllLayerAllTask).abs().cpu(), rate * 100
                    )
                elif pmethod == "mixAR_AllLayerAllTask":
                    threshold = np.percentile(
                        torch.cat(param_AllLayerAllTask).abs().cpu(), rate * 100
                    )
                    if threshold < value:
                        threshold = value
                for task_id in range(1, self.trained_task + 1):
                    reduce_task = []
                    for key in self.WBias_Rperturb_alltask[task_id].keys():
                        paramS = self.WBias_Sperturb_alltask[task_id][key]
                        paramR = self.WBias_Rperturb_alltask[task_id][key]
                        reduce_task.append(
                            round(
                                torch.le(
                                    torch.cat(
                                        (paramS.data.reshape(-1), paramR.data.reshape(-1))
                                    ).abs(),
                                    threshold,
                                )
                                .double()
                                .mean()
                                .item(),
                                2,
                            )
                        )
                        paramS.data *= torch.gt(paramS.data.abs(), threshold)
                        paramR.data *= torch.gt(paramR.data.abs(), threshold)
                    self.log(
                        "Prune task",
                        task_id,
                        "-- each layer reduced by rate:",
                        reduce_task,
                        "by threshold:",
                        round(threshold, 5),
                    )
                return None

            for task_id in range(1, self.trained_task + 1):
                if pmethod in ["relativeAllLayer", "mixAR_AllLayer"]:
                    reduce_task = []
                    param_AllLayer = ()
                    for key in self.WBias_Sperturb_alltask[task_id].keys():
                        paramS = self.WBias_Sperturb_alltask[task_id][key]
                        paramR = self.WBias_Rperturb_alltask[task_id][key]
                        param_AllLayer = param_AllLayer + (
                            paramS.data.reshape(-1),
                            paramR.data.reshape(-1),
                        )

                    if pmethod == "relativeAllLayer":
                        threshold = np.percentile(torch.cat(param_AllLayer).abs().cpu(), rate * 100)
                    elif pmethod == "mixAR_AllLayer":
                        threshold = np.percentile(torch.cat(param_AllLayer).abs().cpu(), rate * 100)
                        if threshold < value:
                            threshold = value

                    for key in self.WBias_Sperturb_alltask[task_id].keys():
                        paramS = self.WBias_Sperturb_alltask[task_id][key]
                        paramR = self.WBias_Rperturb_alltask[task_id][key]
                        reduce_task.append(
                            round(
                                torch.le(
                                    torch.cat(
                                        (paramS.data.reshape(-1), paramR.data.reshape(-1))
                                    ).abs(),
                                    threshold,
                                )
                                .double()
                                .mean()
                                .item(),
                                2,
                            )
                        )
                        paramS.data *= torch.gt(paramS.data.abs(), threshold)
                        paramR.data *= torch.gt(paramR.data.abs(), threshold)
                    self.log(
                        "Prune task",
                        task_id,
                        "-- each layer reduced by rate:",
                        reduce_task,
                        "by threshold:",
                        round(threshold, 5),
                    )

                elif pmethod == "absolute":
                    reduce_task = []
                    for key in self.WBias_Sperturb_alltask[task_id].keys():
                        paramS = self.WBias_Sperturb_alltask[task_id][key]
                        paramR = self.WBias_Rperturb_alltask[task_id][key]
                        threshold = value
                        reduce_task.append(
                            round(
                                torch.le(
                                    torch.cat(
                                        (paramS.data.reshape(-1), paramR.data.reshape(-1))
                                    ).abs(),
                                    threshold,
                                )
                                .double()
                                .mean()
                                .item(),
                                2,
                            )
                        )
                        paramS.data *= torch.gt(paramS.data.abs(), threshold)
                        paramR.data *= torch.gt(paramR.data.abs(), threshold)
                    self.log("Prune task", task_id, "-- each layer reduced by rate:", reduce_task)

                elif pmethod in ["relative", "mixAR"]:
                    threshold_task = []
                    for key in self.WBias_Sperturb_alltask[task_id].keys():
                        paramS = self.WBias_Sperturb_alltask[task_id][key]
                        paramR = self.WBias_Rperturb_alltask[task_id][key]
                        if pmethod == "relative":
                            threshold = np.percentile(
                                torch.cat((paramS.data.reshape(-1), paramR.data.reshape(-1)))
                                .abs()
                                .cpu(),
                                rate * 100,
                            )
                        elif pmethod == "mixAR":
                            threshold = np.percentile(
                                torch.cat((paramS.data.reshape(-1), paramR.data.reshape(-1)))
                                .abs()
                                .cpu(),
                                rate * 100,
                            )
                            if threshold < value:
                                threshold = value

                        threshold_task.append(round(threshold, 5))
                        paramS.data *= torch.gt(paramS.data.abs(), threshold)
                        paramR.data *= torch.gt(paramR.data.abs(), threshold)
                        self.log("Prune task", task_id, "with threshold (5f):", threshold_task)

    def update_model(self, inputs, targets, task_id, regularization=True):
        outputs = self.forward(inputs, task_id)
        loss = self.criterion(outputs, targets, regularization)
        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param_groups in self.optimizer.param_groups:
                for param in param_groups["params"]:
                    # print('!!!!param is', len(param))
                    param.grad *= 10.0
        self.optimizer.step()
        self.scheduler.step()  # pytorch 1.5 revised
        return loss.detach(), outputs.detach()

    def learn_batch(
        self, train_loader, task_id, val_loader=None, preval_loader=None, preval_id=None
    ):
        assert task_id == (self.trained_task + 1), "expect task_id to be {}".format(
            self.trained_task + 1
        )
        if task_id > 0:
            self.log("Model and Optimizer are reset for the new task!")
            self.new_task(task_id)  # Reset model
            self.trained_task += 1
            self.init_optimizer(lowRank=False)
            self.estimate_ranks(train_loader, task_id, self.approxiRate, val_loader)
            self.init_optimizer(lowRank=True)
        elif task_id == 0:
            self.trained_task += 1

        if task_id == 1:
            self.config["n_epochs"] = int(self.config["n_epochs"] - self.config["estRank_epoch"])

        self.update_model_newtask(task_id)

        self.log("Sparse penalty:", self.config["l1_hyp"])
        self.log("Epoch\tBatch\tLoss\ttraAcc\tvalAcc\ttraT\tdataT\tvalT")
        print_freq = self.config["print_freq"]
        epoch_freq = self.config["epoch_freq"]
        acc_max = 0.0
        acc_max_idx = 0

        for epoch in range(self.config["n_epochs"]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            self.model.train()

            data_timer.tic()
            for i, (inputs, targets) in enumerate(train_loader):
                batch_timer.tic()
                data_time.update(data_timer.toc())

                if self.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                loss, outputs = self.update_model(inputs, targets, task_id)
                inputs = inputs.detach()
                targets = targets.detach()

                acc.update(accuracy(outputs, targets), len(targets))
                losses.update(loss, inputs.size(0))

                batch_time.update(batch_timer.toc()) 

                if (epoch == 0 and i == 0) or (
                    (epoch % epoch_freq) == 0
                    and print_freq > 0
                    and (i % print_freq) == (print_freq - 1)
                    and i != (len(train_loader) - 1)
                ):
                    if val_loader != None:
                        val_result = self.validation(val_loader, task_id)
                        val_acc = val_result["acc"]
                        val_time = val_result["time"]

                        self.log(
                            "[{0}/{1}]\t"
                            "[{2}/{3}]\t"
                            "{loss:.2f}\t"
                            "{acc:.2f}\t"
                            "{val:.2f}\t"
                            "{train_time:.1f}s\t"
                            "{data_time:.1f}s\t"
                            "{val_time:.1f}s".format(
                                epoch + 1,
                                self.config["n_epochs"],
                                i + 1,
                                len(train_loader),
                                loss=losses.val,
                                acc=acc.val,
                                val=val_acc,
                                train_time=batch_time.sum,
                                data_time=data_time.sum,
                                val_time=val_time,
                            )
                        )
                    else:
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

            if (epoch % epoch_freq) == 0 or ((epoch + 1) == self.config["n_epochs"]):
                if val_loader != None:
                    val_result = self.validation(val_loader, task_id)
                    val_acc = val_result["acc"]
                    val_time = val_result["time"]

                    if (acc_max < val_acc) and (
                        epoch >= (self.config["n_epochs"] - self.config["check_lepoch"])
                    ):
                        acc_max = val_acc
                        acc_max_idx = epoch
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
                    "{val:.2f}\t"
                    "{train_time:.1f}s\t"
                    "{data_time:.1f}s\t"
                    "{val_time:.1f}s".format(
                        epoch + 1,
                        self.config["n_epochs"],
                        i + 1,
                        len(train_loader),
                        loss=losses.val,
                        acc=acc.val,
                        val=val_acc,
                        train_time=batch_time.sum,
                        data_time=data_time.sum,
                        val_time=val_time,
                    )
                )

        if val_loader != None and (self.config["check_lepoch"] > 0):
            self.log("Retrieve: the best model is at epoch:", acc_max_idx + 1, "with acc:", acc_max)
            self.load_model_task(task_id, acc_max_idx)
        for param in self.model_alltask[str(task_id)].parameters():
            param.requires_grad = False

        if task_id > 0:
            self.l1_pruning(self.config["prune_value"], self.config["prune_method"])
            self.increase_size_rate = round(self.count_parameter() / self.baseSize - 1, 3)
            self.log("Increased parameters so far over base model:", self.increase_size_rate)

    def count_parameter(self, sparse=True):
        totalNum_param = self.baseSize
        for task_id in range(1, self.trained_task + 1):
            for key, param in self.WMask_Sperturb_alltask[task_id].items():
                totalNum_param += param.numel()
            for key, param in self.WMask_Rperturb_alltask[task_id].items():
                totalNum_param += param.numel()
            for key, param in self.WBias_Sperturb_alltask[task_id].items():
                if sparse:
                    totalNum_param += (param.data.abs() > 1e-8).sum().item()
                else:
                    totalNum_param += param.numel()
            for key, param in self.WBias_Rperturb_alltask[task_id].items():
                if sparse:
                    totalNum_param += (param.data.abs() > 1e-8).sum().item()
                else:
                    totalNum_param += param.numel()
            for key, bnLayer in self.BN_alltask[task_id].items():
                for param in bnLayer.parameters():
                    totalNum_param += param.numel()
            for param in self.head_alltask[str(task_id)].parameters():
                totalNum_param += param.numel()

        return totalNum_param

    def validation(self, dataloader, task_id):
        assert task_id <= self.trained_task, "task {} has not been trained yet".format(task_id)
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
                outputs = self.forward(inputs, task_id)
                acc.update(accuracy(outputs, targets), len(targets))
        self.train(orig_mode)
        return {"acc": acc.avg, "time": batch_timer.toc()}

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.l1_hyp = self.l1_hyp.cuda()
        self.wd_rate = self.wd_rate.cuda()
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        self.model_alltask = self.model_alltask.cuda()
        self.head_alltask = self.head_alltask.cuda()

        for key in self.importance_alltask.keys():
            for _, imp in self.importance_alltask[key].items():
                imp = imp.cuda()

        for key in self.BN_alltask.keys():
            self.BN_alltask[key] = self.BN_alltask[key].cuda()

        for key in self.WMask_Sperturb_alltask.keys():
            self.WMask_Sperturb_alltask[key] = self.WMask_Sperturb_alltask[key].cuda()

        for key in self.WMask_Rperturb_alltask.keys():
            self.WMask_Rperturb_alltask[key] = self.WMask_Rperturb_alltask[key].cuda()

        for key in self.WBias_Sperturb_alltask.keys():
            self.WBias_Sperturb_alltask[key] = self.WBias_Sperturb_alltask[key].cuda()

        for key in self.WBias_Rperturb_alltask.keys():
            self.WBias_Rperturb_alltask[key] = self.WBias_Rperturb_alltask[key].cuda()

    def cpu(self):
        self.l1_hyp = self.l1_hyp.cpu()
        self.wd_rate = self.wd_rate.cpu()
        self.model = self.model.cpu()
        self.criterion_fn = self.criterion_fn.cpu()
        self.model_alltask = self.model_alltask.cpu()
        self.head_alltask = self.head_alltask.cpu()

        for key in self.importance_alltask.keys():
            for _, imp in self.importance_alltask[key].items():
                imp = imp.cpu()

        for key in self.BN_alltask.keys():
            self.BN_alltask[key] = self.BN_alltask[key].cpu()

        for key in self.WMask_Sperturb_alltask.keys():
            self.WMask_Sperturb_alltask[key] = self.WMask_Sperturb_alltask[key].cpu()

        for key in self.WMask_Rperturb_alltask.keys():
            self.WMask_Rperturb_alltask[key] = self.WMask_Rperturb_alltask[key].cpu()

        for key in self.WBias_Sperturb_alltask.keys():
            self.WBias_Sperturb_alltask[key] = self.WBias_Sperturb_alltask[key].cpu()

        for key in self.WBias_Rperturb_alltask.keys():
            self.WBias_Rperturb_alltask[key] = self.WBias_Rperturb_alltask[key].cpu()
