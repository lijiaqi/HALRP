import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature, device, base_temperature=0.07, contrast_mode="all"):
        super(SupConLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

        self.device = device

    def forward(self, features, labels=None):
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz,  , ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device) # shape=(N,N)

        contrast_count = features.shape[1] # number of views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (N*V, d)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0] # shape=(N, d)
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature # (N*V, d)
            anchor_count = contrast_count # V
        else:
            raise ValueError("Unknown contrast mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        ) # shape=(N or N*V, N*V)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # (N or N*V, N*V)
        # mask-out self-contrast cases
        ### zero for diagonal (square or rectangular), one for others
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        ### mask.sum(1) can contain zero if contrast_count=1, lead to Nan in loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
