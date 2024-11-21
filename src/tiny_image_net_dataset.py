import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm
from torchvision.datasets.utils import download_url
import wget
import sys

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train", download=False, transform=None):
        self.root = os.path.join(root, split) #  The zip file of Tiny Image Net contains folders for train, val, and test so we can set the root this way
        self.split = split
        self.transform = transform
        self.data = []

        if download:
            self.download()

        if split == "train":
            self._load_train_data()
        elif split == "val" or split == "test":
            self._load_test_val_data()
        else:
            raise ValueError(f"Invalid split: {split}")

    @staticmethod
    def _progress_bar_download(current, total, width=80):
        # https://stackoverflow.com/questions/58125279/python-wget-module-doesnt-show-progress-bar
        progress_message = f"Downloading... {100*current/total:.2f} %"
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    @staticmethod
    def download():
        if os.path.exists("../data/tiny-imagenet-200"):
            return

        if not os.path.exists("../data"):
            os.makedirs("../data")

        if not os.path.exists("../data/tiny-imagenet-200"):
            wget.download("http://cs231n.stanford.edu/tiny-imagenet-200.zip", out="../data/tiny-imagenet-200.zip", bar=TinyImageNetDataset._progress_bar_download)

            with ZipFile("../data/tiny-imagenet-200.zip", "r") as zipfile:
                file_list = zipfile.namelist()
                with tqdm(total=len(file_list), desc="Extracting") as pbar:
                    for file in file_list:
                        zipfile.extract(file, path="../data")
                        pbar.update(1)

            os.remove("../data/tiny-imagenet-200.zip")


    def _load_train_data(self):
        for subdir in os.listdir(self.root):
            subdir_path = os.path.join(self.root, subdir, "images")
            for img_file in os.listdir(subdir_path):
                self.data.append(os.path.join(subdir_path, img_file))
        self.data = np.array(self.data)

    def _load_test_val_data(self):
        images_path = os.path.join(self.root, "images")
        self.data = np.array([os.path.join(images_path, img_file) for img_file in os.listdir(images_path)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        mask = None
        if self.transform:
            image, mask = self.transform(image)
        return image, mask
