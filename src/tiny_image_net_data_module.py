from src.TinyImageNetDataset import TinyImageNetDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# AIIP Exercises & https://lightning.ai/docs/pytorch/stable/data/datamodule.html

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data/tiny-imagenet-200", batch_size_train=32, batch_size_val=32, batch_size_test=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train, self.batch_size_val, self.batch_size_test = batch_size_train, batch_size_val, batch_size_test
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        TinyImageNetDataset.download()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train = TinyImageNetDataset(self.data_dir, split="train")
            self.val = TinyImageNetDataset(self.data_dir, split="val")
        elif stage == "test":
            self.test = TinyImageNetDataset(self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size_train)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size_val)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size_test)
