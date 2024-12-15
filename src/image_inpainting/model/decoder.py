import torch
from torch import nn

class Decoder(nn.Module):
	def __init__(self, latent_dim=4000):
		"""Décodeur qui reconstruit une image de 64x64 à partir d'un vecteur latent."""
		super(Decoder, self).__init__()

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),  # Output: 4x4x512
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
   
			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 8x8x256
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
   
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 16x16x128
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
   
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 32x32x64
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
   
			nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: 64x64x3
			nn.Tanh()  # To normalize output between [-1, 1]
		)

	def forward(self, x):
		return self.decoder(x)

# Test the decoder
if __name__ == "__main__":
	# Create an instance of the Decoder
	decoder = Decoder(latent_dim=4000)
	# Define a dummy latent vector (batch size = 1, latent_dim = 4000, height = 1, width = 1)
	dummy_latent = torch.randn(1, 4000, 1, 1)
	# Pass the dummy latent vector through the Decoder
	reconstructed_image = decoder(dummy_latent)
	# Print the output shape
	print("Latent vector shape:", dummy_latent.shape)  # Expected: torch.Size([1, 4000, 1, 1])
	print("Reconstructed image shape:", reconstructed_image.shape)  # Expected: torch.Size([1, 3, 64, 64])
