import yaml
import torch
from data import get_dataloaders
from model import ProtoNet
from loss import Loss
from train import train, valid
from test import test
from utils import save_checkpoint


def main(config):
    config["device"] = device = "cuda" if torch.cuda.is_available() else "cpu"

    if not test:
        trainloader, validloader = get_dataloaders(config, "trainval", "test")

        model = ProtoNet().to(device)
        crit = Loss().to(device)
        opt = torch.optim.Adam(model.parameters(), config["lr"])

        sched = torch.optim.lr_scheduler.StepLR(
            optimizer=opt,
            gamma=config["lr_sched_gamma"],
            step_size=config["lr_sched_step"],
        )

        best_acc = 0
        for epoch in range(1, config["num_epoch"] + 1):
            loss_train = train(trainloader, model, opt, crit, config)

            is_test = False if epoch % config["num_epi_test"] else True
            if is_test or epoch == config["num_epoch"] or epoch == 1:
                loss_valid, acc = valid(validloader, model, crit, config)

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


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
