import torch
from torch import nn

class Decoder(nn.Module):
	"""
	A Decoder class for image inpainting using a basic convolutional network.
	Attributes:
		decoder (nn.Sequential): A sequential container of transposed convolutional layers, batch normalization layers, 
								 and activation functions to decode the latent representation into an image.
	"""
	def __init__(self, latent_dim=4000):
		"""
		Initializes the Decoder model.
		Args:
			latent_dim (int): The dimensionality of the latent space. Default is 4000.
		The Decoder model consists of a series of ConvTranspose2d layers with BatchNorm2d and ReLU activations,
		followed by a final ConvTranspose2d layer with a Tanh activation to normalize the output between [-1, 1].
		"""
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
		"""
		Perform a forward pass through the decoder.

		Args:
			x (torch.Tensor): The input tensor to be processed by the decoder.

		Returns:
			torch.Tensor: The output tensor after being processed by the decoder.
		"""
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
