import numpy as np
from torch.utils.data import Dataset
import os
import cv2

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = os.path.join(root, split) #  The zip file of Tiny Image Net contains folders for train, val, and test so we can set the root this way
        self.split = split
        self.transform = None # not yet defined in this version, but will be used later on
        self.data = []

        if split == "train":
            self._load_train_data()
        elif split == "val" or split == "test":
            self._load_test_val_data()
        else:
            raise ValueError(f"Invalid split: {split}")

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
        if self.transform:
            image = self.transform(image)
        return image
