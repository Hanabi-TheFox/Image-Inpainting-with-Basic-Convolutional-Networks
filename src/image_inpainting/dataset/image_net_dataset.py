import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

class ImageNetDataset(Dataset):
    """ImageNet dataset for image inpainting
    
    Attributes:
        root (str): Root directory of the dataset
        split (str): Split of the dataset (train, val, test)
        transform (torchvision.transforms): Transform to apply to the images
        data (list): List of image paths
    """
    def __init__(self, root, split="train", initialize_splits=False, transform=None):
        """Initialize the ImageNet dataset
        
        Args:
            root (str): Root directory of the dataset
            split (str): Split of the dataset (train, val, test)
            initialize_splits (bool): Initialize the splits of the dataset
            transform (torchvision.transforms): Transform to apply to the images
        """
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
        """Load the train, test, or val data"""
        for subdir in os.listdir(self.root):
            subdir_path = os.path.join(self.root, subdir)
            for img_file in os.listdir(subdir_path):
                self.data.append(os.path.join(subdir_path, img_file))
        self.data = np.array(self.data)

    def __len__(self):
        """Return the length of the dataset
        
        Returns:
            int: Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get an item from the dataset and apply the needed transformations beforehand
        
        Returns:
            tuple: Image with masked region (dropout center) and the region that was dropped out
        """
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR on RGB
        image = Image.fromarray(image)  # Convert to PIL.Image

        if self.transform:
            image = self.transform(image)  # Transform the image into a resized normalised tensor (3, 128, 128)

        # Crating mask (64x64)
        mask_size = image.shape[1] // 2  # mask size is half of the image size
        center_start = (image.shape[1] // 2 - mask_size // 2, image.shape[2] // 2 - mask_size // 2)
        center_end = (center_start[0] + mask_size, center_start[1] + mask_size)

        # Get the region that will be masked
        masked_region = image[:, center_start[0]:center_end[0], center_start[1]:center_end[1]].clone()

        # Mask the region with a white square
        # As mentioned in section 3.3. Region masks in the paper
        image[:, center_start[0]:center_end[0], center_start[1]:center_end[1]] = 0

        return image, masked_region
