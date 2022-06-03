import torch
from utils import RunningAvg


def train(dataloader, model, optimizer, criterion, config):
    model.train()

    losses = RunningAvg()
    num_spt = config["num_spt_train"]
    for x, y in dataloader:
        x, y = x.to(config["device"]), y.to(config["device"])

        y_pred = model(x)
        loss, _ = criterion(y_pred, y, num_spt, config["device"])

        losses.update(loss.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


@torch.no_grad()
def valid(dataloader, model, criterion, config):
    model.eval()

    losses, accs = RunningAvg(), RunningAvg()
    num_spt = config["num_spt_valid"]
    for x, y in dataloader:
        x, y = x.to(config["device"]), y.to(config["device"])

        y_pred = model(x)
        loss, acc1 = criterion(y_pred, y, num_spt, config["device"])

        losses.update(loss.item(), x.size(0))
        accs.update(acc1.item(), x.size(0))

    return losses.avg, accs.avg
