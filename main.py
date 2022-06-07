import yaml
import torch
from train import train
from test import test


def main():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["device"] = "cuda:3" if torch.cuda.is_available() else "cpu"

    if not config["test"]:
        train(config)
    else:
        test(config)


if __name__ == "__main__":
    main()
