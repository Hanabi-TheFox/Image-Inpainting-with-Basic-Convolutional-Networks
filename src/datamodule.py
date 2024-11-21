from src.TinyImageNetDataset import TinyImageNetDataset
import pytorch_lightning as pl

# AIIP Exercises & https://lightning.ai/docs/pytorch/stable/data/datamodule.html

dataset = TinyImageNetDataset(root='../data/tiny-imagenet-200', split='train', download=True)
print(len(dataset))

# class MNISTDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir="../data/tiny-imagenet-200", batch_size_train=32, batch_size_val=32, batch_size_test=32):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size_train, self.batch_size_val, self.batch_size_test = batch_size_train, batch_size_val, batch_size_test
#
#
#
#     def setup(self, stage: str):
#         self.mnist_test = MNIST(self.data_dir, train=False)
#         self.mnist_predict = MNIST(self.data_dir, train=False)
#         mnist_full = MNIST(self.data_dir, train=True)
#         self.mnist_train, self.mnist_val = random_split(
#             mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
#         )
#
#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=self.batch_size)
#
#     def val_dataloader(self):
#         return DataLoader(self.mnist_val, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=self.batch_size)
#
#     def predict_dataloader(self):
#         return DataLoader(self.mnist_predict, batch_size=self.batch_size)
#
#     def teardown(self, stage: str):
#         # Used to clean-up when the run is finished
#         ...