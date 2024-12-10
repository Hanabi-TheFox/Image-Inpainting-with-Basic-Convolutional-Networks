import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

from image_inpainting.adversarial_discriminator import AdversarialDiscriminator

from image_inpainting.encoder import Encoder
from image_inpainting.decoder import Decoder
from image_inpainting.tiny_image_net_data_module import TinyImageNetDataModule


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
