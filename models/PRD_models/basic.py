import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(
        np.floor(
            (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1
        )
    )


class View(nn.Module):
    def __init__(self, n_feat):
        super(View, self).__init__()
        self.n_feat = n_feat

    def forward(self, x):
        return x.view(-1, self.n_feat)

# Classifiers
def add_linear(dim_in, dim_out, proj_bn, relu):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out))
    if proj_bn:
        layers.append(nn.BatchNorm1d(dim_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return layers


class ProjectionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, feat_dim, proj_bn, num_layers):
        super(ProjectionMLP, self).__init__()

        self.layers = self._make_layers(
            dim_in, hidden_dim, feat_dim, proj_bn, num_layers
        )

    def _make_layers(self, dim_in, hidden_dim, feat_dim, proj_bn, num_layers):
        layers = []
        layers.extend(add_linear(dim_in, hidden_dim, proj_bn=proj_bn, relu=True))

        for _ in range(num_layers - 2):
            layers.extend(
                add_linear(hidden_dim, hidden_dim, proj_bn=proj_bn, relu=True)
            )

        layers.extend(add_linear(hidden_dim, feat_dim, proj_bn=False, relu=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Prototypes(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        n_classes_per_task: int,
        n_tasks: int,
    ):
        super(Prototypes, self).__init__()

        self.heads = self._create_prototypes(
            dim_in=feat_dim,
            n_classes=n_classes_per_task,
            n_heads=n_tasks,
        )

    def _create_prototypes(
        self, dim_in: int, n_classes: int, n_heads: int
    ) -> torch.nn.ModuleDict:

        layers = {}
        for t in range(n_heads):
            layers[str(t)] = nn.Linear(dim_in, n_classes, bias=False)

        return nn.ModuleDict(layers)

    def forward(self, x: torch.FloatTensor, task_id: int) -> torch.FloatTensor:
        out = self.heads[str(task_id)](x)
        return out
