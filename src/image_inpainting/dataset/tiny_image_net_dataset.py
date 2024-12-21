import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from zipfile import ZipFile
from tqdm import tqdm
import wget
import sys
from PIL import Image

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset for image inpainting
    
    Attributes:
        root (str): Root directory of the dataset
        split (str): Split of the dataset (train, val, test)
        transform (torchvision.transforms): Transform to apply to the images
        data (list): List of image paths
    """
    def __init__(self, root, split="train", download=False, transform=None):
        """Initialize the Tiny ImageNet dataset
        
        Args:
            root (str): Root directory of the dataset
            split (str): Split of the dataset (train, val, test)
            download (bool): Download the dataset
            transform (torchvision.transforms): Transform to apply to the images
        """
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
        """Show a progress bar for the download
        
        Args:
            current (int): Current progress
            total (int): Total progress
            width (int): Width of the progress bar
        """
        # https://stackoverflow.com/questions/58125279/python-wget-module-doesnt-show-progress-bar
        progress_message = f"Downloading... {100*current/total:.2f} %"
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    @staticmethod
    def download(path="../../data/tiny-imagenet-200"):
        """Download the Tiny ImageNet dataset
        
        Args:
            path (str): Path to download the dataset
        """
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            return

        if not os.path.exists(path):
            os.makedirs(path)

        path_zip = os.path.join(os.path.dirname(path), "tiny-imagenet-200.zip")

        if not os.path.exists(path_zip):
            wget.download("http://cs231n.stanford.edu/tiny-imagenet-200.zip", out=path_zip, bar=TinyImageNetDataset._progress_bar_download)

        with ZipFile(path_zip, "r") as zipfile:
            file_list = zipfile.namelist()
            with tqdm(total=len(file_list), desc="Extracting...") as pbar:
                for file in file_list:
                    zipfile.extract(file, path=path)
                    pbar.update(1)

        # in tiny-imagenet-200.zip, the content is in tiny-imagenet-200 folder, here we move the content out of the zip root folder
        unzipped_folder = os.path.join(path, "tiny-imagenet-200")
        print("Moving extracted files...")
        for file in tqdm(os.listdir(unzipped_folder)):
            os.rename(os.path.join(unzipped_folder, file), os.path.join(path, file))
        os.rmdir(os.path.join(unzipped_folder))

        os.remove(path_zip)


    def _load_train_data(self):
        """Load the train data"""
        for subdir in os.listdir(self.root):
            subdir_path = os.path.join(self.root, subdir, "images")
            for img_file in os.listdir(subdir_path):
                self.data.append(os.path.join(subdir_path, img_file))
        self.data = np.array(self.data)

    def _load_test_val_data(self):
        """Load the test or val data"""
        images_path = os.path.join(self.root, "images")
        self.data = np.array([os.path.join(images_path, img_file) for img_file in os.listdir(images_path)])

    def __len__(self):
        """Return the length of the dataset
        
        Returns:
            int: Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get an item from the dataset and apply the needed transformations beforehand
        
        Args:
            idx (int): Index of the item to get
        
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
