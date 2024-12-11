import torch
from torch import nn

class Encoder(nn.Module):
	def __init__(self, input_channels=3, latent_dim=4000):
		super(Encoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 32x32x64
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 16x16x128
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 8x8x256
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: 4x4x512
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, latent_dim, kernel_size=4, stride=1, padding=0),  # Output: 1x1xlatent_dim
			nn.ReLU(inplace=True)
		)

		# Fully connected channel-wise layer
		self.channel_fc = nn.Sequential(
			nn.Flatten(),  # Flatten to [batch_size, latent_dim]
			nn.Linear(latent_dim, latent_dim),  # Fully connected across channels
			nn.ReLU(inplace=True),
			nn.Unflatten(1, (latent_dim, 1, 1))  # Reshape to [batch_size, latent_dim, 1, 1]
		)

	def forward(self, x):
		features = self.encoder(x)  # Extract features
		latent = self.channel_fc(features)  # Fully connected channel-wise
		return latent

# Test the encoder
if __name__ == "__main__":
	# Create an instance of the Encoder
	encoder = Encoder(input_channels=3, latent_dim=4000)
	# Define a dummy input tensor (batch size = 1, channels = 3, height = 128, width = 128)
	dummy_input = torch.randn(1, 3, 128, 128)
	# Pass the dummy input through the Encoder
	latent_vector = encoder(dummy_input)
	# Print the output shape
	print("Input shape:", dummy_input.shape)  # Expected: torch.Size([1, 3, 128, 128])
	print("Latent vector shape:", latent_vector.shape)  # Expected: torch.Size([1, 4000, 1, 1])

