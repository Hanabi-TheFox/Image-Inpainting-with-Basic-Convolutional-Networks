from torch import nn

from image_inpainting.model.encoder import Encoder
from image_inpainting.model.decoder import Decoder


class AdversarialGenerator(nn.Module):
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000):
        super(AdversarialGenerator, self).__init__()

        # Define the encoder, the decoder and the discriminator
        self.encoder = Encoder(input_channels=input_size[0], latent_dim=hidden_size)
        self.decoder = Decoder(latent_dim=hidden_size)

    def forward(self, x):
        """Forward pass: encoder + decoder"""
        latent = self.encoder(x)  # Encode the image
        reconstruction = self.decoder(latent)  # Decode to reconstruct the image
        return reconstruction
