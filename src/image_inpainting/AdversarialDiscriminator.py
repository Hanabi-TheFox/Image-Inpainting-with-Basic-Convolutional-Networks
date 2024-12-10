import torch
from torch import nn

class AdversarialDiscriminator(nn.Module):
	def __init__(self, input_channels=3):
		super(AdversarialDiscriminator, self).__init__()

		self.discriminator = nn.Sequential(
			nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # Output: 32x32x64
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 16x16x128
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 8x8x256
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: 4x4x512
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output: 1x1x1
			nn.Sigmoid()  # Output between 0 and 1 (real or fake)
		)

	def forward(self, x):
		return self.discriminator(x).view(-1, 1)  # Flatten output for binary classification

# Test the Discriminator
if __name__ == "__main__":
	discriminator = AdversarialDiscriminator(input_channels=3)
	dummy_input = torch.randn(1, 3, 64, 64)  # Example input image
	output = discriminator(dummy_input)
	print("Input shape:", dummy_input.shape)  # Expected: torch.Size([1, 3, 64, 64])
	print("Output shape:", output.shape)  # Expected: torch.Size([1, 1])
