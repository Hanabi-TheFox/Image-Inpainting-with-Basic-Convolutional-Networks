import torch

from image_inpainting.image_net_dataset import ImageNetDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# AIIP Exercises & https://lightning.ai/docs/pytorch/stable/data/datamodule.html

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../../data/imagenet", batch_size_train=32, batch_size_val=32, batch_size_test=32, num_workers=0, pin_memory=False, persistent_workers=False):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
                            transforms.Resize((128, 128)),  # Redimensionner l'image en 64x64
                            transforms.ToTensor(),  # Convertir l'image en tenseur
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image, values often taken for ImageNet
                        ])
        self.batch_size_train, self.batch_size_val, self.batch_size_test = batch_size_train, batch_size_val, batch_size_test
        self.train = None
        self.val = None
        self.test = None
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        print("Note: The ImageNet dataset can't be downloaded automatically. Please refer to the README if you haven't already downloaded it")

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train = ImageNetDataset(self.data_dir, split="train", transform=self.transform)
            self.val = ImageNetDataset(self.data_dir, split="val", transform=self.transform)
        elif stage == "test":
            self.test = ImageNetDataset(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size_train, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size_val, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

if __name__ == '__main__':
    data_module = ImageNetDataModule()
    data_module.prepare_data()
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    for x, y in val_loader:
        print(x.shape, y.shape)
        break
    for x, y in test_loader:
        print(x.shape, y.shape)
        break

    for img, mask in train_loader:
        grid_img = make_grid(img[:4])
        grid_mask = make_grid(mask[:4])
        # dénomarliser les images pour l'affichage
        grid_img = torch.clamp((grid_img - grid_img.min()) / (grid_img.max() - grid_img.min()), 0, 1)
        grid_mask = torch.clamp((grid_mask - grid_mask.min()) / (grid_mask.max() - grid_mask.min()), 0, 1)

        plt.figure(figsize=(8, 10))
        plt.subplot(2, 1, 1)
        plt.title("Images Masquées")
        plt.imshow(grid_img.permute(1, 2, 0).numpy())
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.title("Masques")
        plt.imshow(grid_mask.permute(1, 2, 0).numpy())
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        break