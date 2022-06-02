import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calc_distance


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, true, num_spt, device):
        cls_ = torch.unique(true)
        num_cls = len(cls_)

        # computes barycenters
        idx_spt = torch.stack(
            list(map(lambda c: true.eq(c).nonzero()[:num_spt].squeeze(1), cls_))
        )
        protos = torch.stack([data[idx].mean(0) for idx in idx_spt])

        # computes distances
        num_qry = true.eq(cls_[0].item()).sum().item() - num_spt
        idx_qry = torch.stack(
            list(map(lambda c: true.eq(c).nonzero()[num_spt:], cls_))
        ).view(-1)
        samples_qry = data[idx_qry]
        dists = calc_distance(samples_qry, protos, "l2")

        # predicts
        log_prob = F.log_softmax(-dists, dim=1)
        pred = log_prob.argmax(1)

        true = torch.arange(0, num_cls, 1 / num_qry).long().to(device)

        # computes losses and accuracies
        losses = torch.nn.NLLLoss()(log_prob, true)
        accs = pred.eq(true.squeeze()).float().mean()
        return losses, accs
