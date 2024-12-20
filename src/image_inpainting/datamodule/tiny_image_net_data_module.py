import torch
from image_inpainting.dataset.tiny_image_net_dataset import TinyImageNetDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import os
from pathlib import Path
from image_inpainting.utils import insert_image_center

# AIIP Exercises & https://lightning.ai/docs/pytorch/stable/data/datamodule.html

class TinyImageNetDataModule(pl.LightningDataModule):
    """DataModule for Tiny ImageNet dataset.
    
    Attributes:
        data_dir (str): Path to the ImageNet dataset directory.
        batch_size_train (int): Batch size for training dataloader.
        batch_size_val (int): Batch size for validation dataloader.
        batch_size_test (int): Batch size for test dataloader.
        num_workers (int): Number of workers to use for data loading.
        pin_memory (bool): Whether to pin memory for data loading.
        persistent_workers (bool): Whether to keep workers alive between epochs.
    """
    def __init__(self, data_dir="../../data/tiny-imagenet-200", batch_size_train=32, batch_size_val=32, batch_size_test=32, num_workers=0, pin_memory=False, persistent_workers=False):
        """Initializes TinyImageNetDataModule.
        
        Args:
            data_dir (str, optional): Path to the TinyImageNet dataset directory. Defaults to "../../data/tiny-imagenet-200".
            batch_size_train (int, optional): Batch size for training dataloader. Defaults to 32.
            batch_size_val (int, optional): Batch size for validation dataloader. Defaults to 32.
            batch_size_test (int, optional): Batch size for test dataloader. Defaults to 32.
            num_workers (int, optional): Number of workers to use for data loading. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory for data loading. Defaults to False.
            persistent_workers (bool, optional): Whether to keep workers alive between epochs. Defaults to False.
        """
        super().__init__()
        self.data_dir = data_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), # Resize image to 128x128
            transforms.ToTensor(), # Convert image to tensor (normalized to [0,1])
            transforms.Normalize(mean=0.5, std=0.5) # Normalize to -1 1 (because the output is tanh in the generator)
        ])
        
        self.batch_size_train, self.batch_size_val, self.batch_size_test = batch_size_train, batch_size_val, batch_size_test
        self.train = None
        self.val = None
        self.test = None
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def inverse_transform(self, x):
        """Inverse transform the input tensor to an image (with values from 0 to 255).
        
        Args:
            x (torch.Tensor): Input tensor to be transformed.
        
        Returns:
            np.ndarray: Inverse transformed
        """    
        inverse_transformed_x = x * 0.5 + 0.5 # denormalize to [0, 1]
        inverse_transformed_x = inverse_transformed_x.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C) 
        inverse_transformed_x = (inverse_transformed_x * 255).astype(int)
        
        return inverse_transformed_x
        
    def prepare_data(self):
        """Downloads the Tiny ImageNet dataset."""
        TinyImageNetDataset.download(self.data_dir)

    def setup(self, stage: str):
        """Setups the data for the given stage.
        
        Args:
            stage (str): Stage for which to set up the data. Can be "fit" (for training and validation) or "test" (for testing).
        """
        if stage == "fit" or stage is None:
            self.train = TinyImageNetDataset(self.data_dir, split="train", transform=self.transform)
            self.val = TinyImageNetDataset(self.data_dir, split="val", transform=self.transform)
        elif stage == "test":
            self.test = TinyImageNetDataset(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        """Returns the training dataloader.
        
        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(self.train, batch_size=self.batch_size_train, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        """Returns the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(self.val, batch_size=self.batch_size_val, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        """Returns the test dataloader.
        
        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(self.test, batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

# if __name__ == '__main__':
#     dataset_path = os.path.join(Path(__file__).resolve().parent.parent.parent, "data", "tiny-imagenet-200")
     
#     data_module = TinyImageNetDataModule(data_dir=dataset_path)
#     data_module.prepare_data()
#     data_module.setup("fit")
#     train_loader = data_module.train_dataloader()
#     val_loader = data_module.val_dataloader()
#     data_module.setup("test")
#     test_loader = data_module.test_dataloader()

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

#     for img, mask in train_loader:
#         selected_imgs = []
#         selected_masks = []
#         reconstructed_imgs = []
        
#         for i in range(4):
#             selected_imgs.append(data_module.inverse_transform(img[i]))
#             selected_masks.append(data_module.inverse_transform(mask[i]))            
#             reconstructed_imgs.append(insert_image_center(selected_imgs[i], selected_masks[i]))
                
#         axes = plt.figure(figsize=(12, 12))
        
        
#         for i in range(4):
#             plt.subplot(4, 3, 3*i + 1)
#             plt.imshow(selected_imgs[i])
#             plt.axis("off")
#             plt.subplot(4, 3, 3*i + 2)
#             plt.imshow(selected_masks[i])
#             plt.axis("off")
#             plt.subplot(4, 3, 3*i + 3)
#             plt.imshow(reconstructed_imgs[i])
#             plt.axis("off")
        
#         plt.tight_layout()
#         plt.show()
#         break