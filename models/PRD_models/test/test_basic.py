import torch
import torch.nn as nn

def test_ProjectionMLP():
    from PRD_models.basic import ProjectionMLP
    projection = ProjectionMLP(
        dim_in=256, hidden_dim=512, feat_dim=128, proj_bn=False, num_layers=2
    )
    print(projection)

    projection = ProjectionMLP(
        dim_in=256, hidden_dim=512, feat_dim=128, proj_bn=False, num_layers=3
    )
    print(projection)

def test_ProjectionMLP_1layer():
    from PRD_models.basic import ProjectionMLP
    projection = ProjectionMLP(
        dim_in=256, hidden_dim=512, feat_dim=128, proj_bn=False, num_layers=1
    )
    print(projection)
