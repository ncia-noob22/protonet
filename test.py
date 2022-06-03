import torch
from utils import RunningAvg


@torch.no_grad()
def test(dataloader, model, criterion, config):
    model.eval()

    losses, accs = RunningAvg(), RunningAvg()
    num_spt = config["num_spt_test"]
    for x, y in dataloader:
        x, y = x.to(config["device"]), y.to(config["device"])

        y_pred = model(x)
        loss, acc1 = criterion(y_pred, y, num_spt, config["device"])

        losses.update(loss.item(), x.size(0))
        accs.update(acc1.item(), x.size(0))

    return losses.avg, accs.avg
