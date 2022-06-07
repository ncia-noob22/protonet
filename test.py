import torch
from data import get_dataloaders
from model import ProtoNet
from loss import Loss
from utils import RunningAvg


@torch.no_grad()
def _test(dataloader, model, criterion, config):
    model.eval()

    accs = RunningAvg()
    num_spt = config["num_spt_test"]
    for x, y in dataloader:
        x, y = x.to(config["device"]), y.to(config["device"])

        y_pred = model(x)
        _, acc1 = criterion(y_pred, y, num_spt, config["device"])

        accs.update(acc1.item(), x.size(0))

    return accs.avg


def test(config):
    if not test:
        testloader = get_dataloaders(config)

        model = ProtoNet().to(config["device"])
        crit = Loss().to(config["device"])

        acc = _test(testloader, model, crit, config)
        print(f"accuracy: {acc:.3f}")
