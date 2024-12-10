import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
from image_inpainting.loss import JointLoss
from image_inpainting.Encoder import Encoder
from image_inpainting.Decoder import Decoder

class ContextEncoder(pl.LightningModule):
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000, num_classes=10):
        super(ContextEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.loss_function = JointLoss()

        # Define the encoder
        self.encoder = Encoder(input_channels=input_size[0], latent_dim=hidden_size)

        # Placeholder for decoder (to be implemented later)
        self.decoder = Decoder(latent_dim=hidden_size)


    def forward(self, x):
        """Forward pass: encoder + decoder"""
        latent = self.encoder(x)  # Encode the image
        reconstruction = self.decoder(latent)  # Decode to reconstruct the image
        return reconstruction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # TODO or alternating SGD ???

        return optimizer

    def training_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.forward(x)
        loss = self.loss_function(outputs, y)
        psnr = self.psnr_metric(outputs, y)

        # TODO: might be useful (https://pytorch-lightning.readthedocs.io/en/0.10.0/introduction_guide.html#training-overrides)

        self.log('train_acc', psnr, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        loss_function = nn.CrossEntropyLoss()

        outputs = self.forward(x)
        loss = loss_function(outputs, y)
        psnr = self.psnr_metric(outputs, y)

        self.log('val_acc', psnr, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        loss_function = nn.CrossEntropyLoss()

        outputs = self.forward(x)
        loss = loss_function(outputs, y)

        self.psnr += torchmetrics.functional.accuracy(outputs, y, task="multiclass", num_classes=self.num_classes)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_psnr', self.psnr / (batch_idx+1), prog_bar=True)

    def on_test_epoch_start(self):
        self.psnr = 0

    def on_test_epoch_end(self):
        self.psnr /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
        self.log('Final PSNR', self.psnr)

# Test the ContextEncoder
if __name__ == "__main__":
    # Create an instance of the ContextEncoder
    model = ContextEncoder(input_size=(3, 128, 128), hidden_size=4000, num_classes=10)
    # Define a dummy input tensor (batch size = 1, channels = 3, height = 128, width = 128)
    dummy_input = torch.randn(1, 3, 128, 128)
    # Pass the dummy input through the model
    reconstructed_image = model(dummy_input)
    # Print the output shape
    print("Input shape:", dummy_input.shape)  # Expected: torch.Size([1, 3, 128, 128])
    print("Reconstructed image shape:", reconstructed_image.shape)  # Expected: torch.Size([1, 3, 128, 128])
