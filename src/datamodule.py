from torchvision.datasets import ImageNet

from src.TinyImageNetDataset import TinyImageNetDataset

dataset = TinyImageNetDataset(root='../data/tiny-imagenet-200', split='train')
print(len(dataset))

