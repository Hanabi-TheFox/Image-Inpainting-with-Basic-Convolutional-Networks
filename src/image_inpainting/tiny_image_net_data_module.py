import torch
from image_inpainting.tiny_image_net_dataset import TinyImageNetDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from image_inpainting.add_center_square_transform import AddCenterSquareTransform

# AIIP Exercises & https://lightning.ai/docs/pytorch/stable/data/datamodule.html

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data/tiny-imagenet-200", batch_size_train=32, batch_size_val=32, batch_size_test=32, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
                            AddCenterSquareTransform(),  # Appliquer la transformation personnalisée
                            transforms.ToTensor()  # Convertir l'image en tenseur
                        ])
        self.batch_size_train, self.batch_size_val, self.batch_size_test = batch_size_train, batch_size_val, batch_size_test
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        TinyImageNetDataset.download()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train = TinyImageNetDataset(self.data_dir, split="train", transform=self.transform)
            self.val = TinyImageNetDataset(self.data_dir, split="val", transform=self.transform)
        elif stage == "test":
            self.test = TinyImageNetDataset(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size_train)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size_val)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size_test)

# if __name__ == '__main__':
#     data_module = TinyImageNetDataModule()
#     data_module.prepare_data()
#     data_module.setup("fit")
#     train_loader = data_module.train_dataloader()
#     val_loader = data_module.val_dataloader()
#     data_module.setup("test")
#     test_loader = data_module.test_dataloader()
#
#     print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
#     for x, y in train_loader:
#         print(x.shape, y.shape)
#         break
#     for x, y in val_loader:
#         print(x.shape, y.shape)
#         break
#     for x, y in test_loader:
#         print(x.shape, y.shape)
#         break
