import torch
from data import get_dataloaders
from model import ProtoNet
from loss import Loss
from utils import RunningAvg, save_checkpoint


def _train(dataloader, model, optimizer, criterion, config):
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
def _valid(dataloader, model, criterion, config):
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


def train(config):
    trainloader, validloader = get_dataloaders(config)

    model = ProtoNet().to(config["device"])
    crit = Loss().to(config["device"])
    opt = torch.optim.Adam(model.parameters(), config["lr"])

    sched = torch.optim.lr_scheduler.StepLR(
        optimizer=opt,
        gamma=config["lr_sched_gamma"],
        step_size=config["lr_sched_step"],
    )

    best_acc = 0
    for epoch in range(1, config["num_epoch"] + 1):
        loss_train = _train(trainloader, model, opt, crit, config)

        is_test = False if epoch % config["num_epi_test"] else True
        if is_test or epoch == config["num_epoch"] or epoch == 1:
            loss_valid, acc = _valid(validloader, model, crit, config)

            is_best = False
            if acc > best_acc:
                is_best, best_acc = True, acc

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_acc1": best_acc,
                    "optimizer_state_dict": opt.state_dict(),
                },
                is_best,
                "ckpt",
            )

            print(
                f"[{epoch}/{config['num_epoch']}] train loss {loss_train:.3f}, valid loss {loss_valid:.3f}, acc1 {acc:.3f}, # best_acc {best_acc:.3f}"
            )

        else:
            print(f"[{epoch}/{config['num_epoch']}] train loss {loss_train:.3f}")

        sched.step()
