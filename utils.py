import os
import torch


def calc_distance(x, y, metric):
    """Calculates pairwise distance"""
    num_x = x.shape[0]
    num_y = y.shape[0]

    if metric.lower() in ("l2", "euclid"):
        distances = (
            (
                x.unsqueeze(1).expand(num_x, num_y, -1)
                - y.unsqueeze(0).expand(num_x, num_y, -1)
            )
            .pow(2)
            .sum(dim=2)
        )
        return distances

    elif metric.lower().startswith("cos"):
        x_normalized = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        y_normalized = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        x_expanded = x_normalized.unsqueeze(1).expand(num_x, num_y, -1)
        y_expanded = y_normalized.unsqueeze(0).expand(num_x, num_y, -1)
        cos_sim = (x_expanded * y_expanded).sum(dim=2)
        return 1 - cos_sim

    elif metric.lower().startswith("dot"):
        x_expanded = x.unsqueeze(1).expand(num_x, num_y, -1)
        y_expanded = y.unsqueeze(0).expand(num_x, num_y, -1)

        dot_sim = (x_expanded * y_expanded).sum(dim=2)
        return -dot_sim


class RunningAvg:
    """Class computing running average"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, log_dir):
    ckpt_path = os.path.join(log_dir, f"checkpoint_{state['epoch']}.pth")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if is_best:
        ckpt_path = os.path.join(log_dir, "best_model.pth")
        torch.save(state, ckpt_path)
    else:
        torch.save(state, ckpt_path)
