import os, glob
import shutil
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import Omniglot
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tqdm

DIR_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def get_dataloaders(config, *types):
    use_pin_mem = config["device"].startswith("cuda")

    res = []
    for type in types:
        mdb_path = os.path.join(DIR_DATA, "mdb", f"{type}.mdb")
        if os.path.exists(mdb_path):
            data = torch.load(mdb_path)
        else:
            data = CustomDataset(type)
            if not os.path.exists(os.path.dirname(mdb_path)):
                os.makedirs(os.path.dirname(mdb_path))
            torch.save(data, mdb_path)

        num_cls = config[f"num_cls_{type}"]
        num_spt = config[f"num_spt_{type}"]
        num_qry = config[f"num_qry_{type}"]
        num_epi = config[f"num_epi_{type}"]

        sampler = CustomSampler(data.y, num_cls, num_spt, num_qry, num_epi)
        data_loader = DataLoader(data, batch_sampler=sampler, pin_memory=use_pin_mem)
        res.append(data_loader)
    return res


class CustomDataset(Dataset):
    def __init__(self, type):
        super().__init__()
        self.raw_dir = os.path.join(DIR_DATA, "raw")
        self.split_dir = os.path.join(DIR_DATA, "split")
        self.orig_dir = os.path.join(DIR_DATA, "omniglot-py")
        self.proc_dir = DIR_DATA

        if not os.path.exists(self.raw_dir):
            print("Data not found. Downloading data")
            self.download()

        self.x, self.y = self.make_dataset(type)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        Omniglot(root=DIR_DATA, background=False, download=True)
        Omniglot(root=DIR_DATA, background=True, download=True)

        if not os.path.exists(self.proc_dir):
            os.mkdir(self.proc_dir)

        for p in ["images_background", "images_evaluation"]:
            for f in os.listdir(os.path.join(self.orig_dir, p)):
                shutil.move(os.path.join(self.orig_dir, p, f), self.proc_dir)

        shutil.rmtree(self.orig_dir)

    def make_dataset(self, type):
        with open(os.path.join(self.split_dir, f"{type}.txt"), "r") as f:
            classes = f.read().splitlines()

        x, y = [], []
        for idx, cls_ in enumerate(tqdm(classes, desc="Making dataset")):
            dir_cls, degree = cls_.rsplit("/", 1)
            degree = int(degree[3:])

            transform = A.Compose(
                [
                    A.Resize(28, 28),
                    A.Rotate((degree, degree), p=1),
                    A.Normalize(mean=0.92206, std=0.08426),
                    ToTensorV2(),
                ]
            )

            for img_path in glob(os.path.join(self.raw_dir, dir_cls, "*")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = transform(image=img)["image"]

                x.append(img)
                y.append(idx)
        y = torch.LongTensor(y)
        return x, y


class CustomSampler(Sampler):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return

    def __len__(self):
        return len(self)
