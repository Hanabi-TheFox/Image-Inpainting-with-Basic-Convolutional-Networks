from torch import nn

from image_inpainting.model.encoder import Encoder
from image_inpainting.model.decoder import Decoder


class AdversarialGenerator(nn.Module):
    """
    Adversarial Generator model for image inpainting using a basic convolutional network.
    This model consists of an encoder and a decoder. The encoder compresses the input image into a latent representation,
    and the decoder reconstructs the image from this latent representation.
    Attributes:
        encoder (Encoder): The encoder part of the model which compresses the input image.
        decoder (Decoder): The decoder part of the model which reconstructs the image from the latent representation.
    """
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000):
        """
        Initializes the AdversarialGenerator model.
        Args:
            input_size (tuple, optional): The size of the input image in the format (channels, height, width). 
                                          Default is (3, 128, 128).
            hidden_size (int, optional): The size of the latent space. Default is 4000.
        """
        super(AdversarialGenerator, self).__init__()

        # Define the encoder, the decoder and the discriminator
        self.encoder = Encoder(input_channels=input_size[0], latent_dim=hidden_size)
        self.decoder = Decoder(latent_dim=hidden_size)

    def forward(self, x):
        """
        Perform a forward pass through the adversarial generator model.

        Args:
            x (torch.Tensor): Input tensor representing the image to be processed.

        Returns:
            torch.Tensor: Reconstructed image tensor after encoding and decoding.
        """
        latent = self.encoder(x)  # Encode the image
        reconstruction = self.decoder(latent)  # Decode to reconstruct the image
        return reconstruction
