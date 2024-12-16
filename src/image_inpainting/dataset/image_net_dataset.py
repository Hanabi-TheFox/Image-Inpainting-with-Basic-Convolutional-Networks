import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

class ImageNetDataset(Dataset):
    def __init__(self, root, split="train", initialize_splits=False, transform=None):
        self.root = os.path.join(root, split)
        self.split = split
        self.transform = transform
        self.data = []

        if initialize_splits:
            self.create_splits()

        if split == "train" or split == "val" or split == "test":
            self._load_train_test_val_data()
        else:
            raise ValueError(f"Invalid split: {split}")

    def _load_train_test_val_data(self):
        for subdir in os.listdir(self.root):
            subdir_path = os.path.join(self.root, subdir)
            for img_file in os.listdir(subdir_path):
                self.data.append(os.path.join(subdir_path, img_file))
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR en RGB
        image = Image.fromarray(image)  # Convertir en PIL.Image

        if self.transform:
            image = self.transform(image)  # Transformer l'image en tenseur (3, 128, 128)

        # Créer un masque central (64x64)
        mask_size = image.shape[1] // 2  # Suppose une image carrée H x W
        center_start = (image.shape[1] // 2 - mask_size // 2, image.shape[2] // 2 - mask_size // 2)
        center_end = (center_start[0] + mask_size, center_start[1] + mask_size)

        # Extraire la région masquée (32x32)
        masked_region = image[:, center_start[0]:center_end[0], center_start[1]:center_end[1]].clone()

        # Remplir la région masquée avec 0 (blanc dans [0,1])
        image[:, center_start[0]:center_end[0], center_start[1]:center_end[1]] = 0

        return image, masked_region