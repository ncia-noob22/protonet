import os
from glob import glob
import shutil
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import Omniglot
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def get_dataloaders(config):
    use_pin_mem = config["device"].startswith("cuda")

    if not config["test"]:
        loaders = []
        for type in ["trainval", "test"]:
            mdb_path = os.path.join(DATA_DIR, "mdb", f"{type}.mdb")
            if os.path.exists(mdb_path):
                data = torch.load(mdb_path)
            else:
                data = CustomDataset(type)
                if not os.path.exists(os.path.dirname(mdb_path)):
                    os.makedirs(os.path.dirname(mdb_path))
                torch.save(data, mdb_path)

            type = "train" if type.startswith("train") else "valid"
            num_cls = config[f"num_cls_{type}"]
            num_spt = config[f"num_spt_{type}"]
            num_qry = config[f"num_qry_{type}"]
            num_epi = config[f"num_epi_{type}"]

            sampler = CustomSampler(data.y, num_cls, num_spt, num_qry, num_epi)
            data_loader = DataLoader(
                data, batch_sampler=sampler, pin_memory=use_pin_mem
            )
            loaders.append(data_loader)
        return loaders
    else:
        data = CustomDataset("test")  #! need to adjust
        test_loader = DataLoader(data, pin_memory=use_pin_mem)
        return test_loader


class CustomDataset(Dataset):
    def __init__(self, type):
        super().__init__()
        self.split_dir = os.path.join(DATA_DIR, "split")
        self.orig_dir = os.path.join(DATA_DIR, "omniglot-py")
        self.proc_dir = os.path.join(DATA_DIR, "raw")

        if not os.path.exists(self.proc_dir):
            self.download()
        
        self.x, self.y = self.make_dataset(type)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def download(self):
        Omniglot(root=DATA_DIR, background=False, download=True)
        Omniglot(root=DATA_DIR, background=True, download=True)

        if not os.path.exists(self.proc_dir):
            os.mkdir(self.proc_dir)

        for dirname in ["images_background", "images_evaluation"]:
            for filename in os.listdir(os.path.join(self.orig_dir, dirname)):
                try:
                    shutil.move(
                        os.path.join(self.orig_dir, dirname, filename), self.proc_dir
                    )
                except:
                    pass

        shutil.rmtree(self.orig_dir)

    def make_dataset(self, type):
        with open(os.path.join(self.split_dir, f"{type}.txt"), "r") as f:
            clss = f.read().splitlines()

        x, y = [], []
        for idx, cls_ in enumerate(tqdm(clss, desc="Making dataset")):
            cls_dir, degree = cls_.rsplit("/", 1)
            degree = int(degree[3:])

            transform = A.Compose(
                [
                    A.Resize(28, 28),
                    A.Rotate((degree, degree), p=1),
                    A.Normalize(mean=0.92206, std=0.08426),
                    ToTensorV2(),
                ]
            )

            for img_path in glob(os.path.join(self.proc_dir, cls_dir, "*")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = transform(image=img)["image"]

                x.append(img)
                y.append(idx)
        y = torch.LongTensor(y)
        return x, y


class CustomSampler(Sampler):
    def __init__(
        self,
        labels,
        num_cls,
        num_spt,
        num_qry,
        num_epi,
        data_source=None,
    ):
        """
        Args:
            labels
            num_cls: Number of classes for each episode
            num_cpt, num_qry: Number of samples for each class for each episode
            num_epi: Number of episodes for each epoch
        """
        super().__init__(data_source)
        self.labels = labels
        self.num_cls = num_cls
        self.num_spt = num_spt
        self.num_qry = num_qry
        self.num_epi = num_epi

        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.labels))
        self.indexes = torch.Tensor(
            np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        )
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[
                label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]
            ] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        for _ in range(self.num_epi):
            batch_spt = torch.LongTensor(self.num_spt * self.num_cls)
            batch_qry = torch.LongTensor(self.num_qry * self.num_cls)
            cls_idx = torch.randperm(len(self.classes))[: self.num_cls]
            for i, cls_ in enumerate(self.classes[cls_idx]):
                s_s = slice(i * self.num_spt, (i + 1) * self.num_spt)
                s_q = slice(i * self.num_qry, (i + 1) * self.num_qry)

                label_idx = (
                    torch.arange(len(self.classes)).long()[self.classes == cls_].item()
                )
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[
                    : self.num_spt + self.num_qry
                ]

                batch_spt[s_s] = self.indexes[label_idx][sample_idxs][: self.num_spt]
                batch_qry[s_q] = self.indexes[label_idx][sample_idxs][self.num_spt :]

            batch = torch.cat((batch_spt, batch_qry))
            yield batch

    def __len__(self):
        return self.num_epi
