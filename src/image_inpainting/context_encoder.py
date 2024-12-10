import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

from image_inpainting.adversarial_discriminator import AdversarialDiscriminator
from image_inpainting.adversarial_generator import AdversarialGenerator
from image_inpainting.encoder import Encoder
from image_inpainting.decoder import Decoder
from image_inpainting.loss import JointLoss
from image_inpainting.tiny_image_net_data_module import TinyImageNetDataModule

# Based on https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
class ContextEncoder(pl.LightningModule):
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000, reconstruction_loss_weight = 0.999, adversarial_loss_weight = 0.001):
        super(ContextEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.loss_function = JointLoss(reconstruction_loss_weight=reconstruction_loss_weight, adversarial_loss_weight=adversarial_loss_weight)

        self.test_psnr = 0

        self.generator = AdversarialGenerator(input_size=input_size, hidden_size=hidden_size)
        self.discriminator = AdversarialDiscriminator(input_channels=input_size[0])


    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        # They use a higher learning rate for context encoder (10 times) to that of adversarial discriminator (page 6 of the paper)
        generator_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        return [generator_optimizer, discriminator_optimizer], [] # [] because there is no scheduler

    def training_step(self, batch, batch_idx):
        img, unmasked_image = batch
        outputs = self.forward(x)
        loss_val = self.loss_function(outputs, y) # joint loss
        psnr_val = self.psnr_metric(outputs, y)

        generator_optimizer, discriminator_optimizer = self.get_optimizers()

        # Generator
        self.toggle_optimizer(generator_optimizer) # we don't want to update the discriminator here
        self.manual_backward(loss_val)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)

        # Discriminator
        self.toggle_optimizer(discriminator_optimizer)  # we don't want to update the generator here
        self.manual_backward(loss_val)
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        self.log('train_acc', psnr_val, prog_bar=True)
        self.log('train_loss', loss_val, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        pass
        # x,y = batch
        # loss_function = nn.CrossEntropyLoss()
        #
        # outputs = self.forward(x)
        # loss = loss_function(outputs, y)
        # psnr = self.psnr_metric(outputs, y)
        #
        # self.log('val_acc', psnr, prog_bar=True)
        # self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass
        # x,y = batch
        # loss_function = nn.CrossEntropyLoss()
        #
        # outputs = self.forward(x)
        # loss = loss_function(outputs, y)
        #
        # self.test_psnr += self.psnr_metric(outputs, y)
        #
        # self.log('test_loss', loss, prog_bar=True)
        # self.log('test_psnr', self.test_psnr / (batch_idx+1), prog_bar=True)

    def on_test_epoch_start(self):
        self.test_psnr = 0

    def on_test_epoch_end(self):
        self.test_psnr /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
        self.log('Final PSNR', self.test_psnr)


import matplotlib.pyplot as plt
from pytorch_lightning import loggers as pl_loggers

# Test the ContextEncoder
if __name__ == "__main__":
    # Create an instance of the ContextEncoder
    model = ContextEncoder(input_size=(3, 128, 128), hidden_size=4000)

    # Training
    dm = TinyImageNetDataModule()

    tb_logger = pl_loggers.TensorBoardLogger("Context Encoder - Inpating")
    trainer = pl.Trainer(max_epochs=20, devices=-1, accelerator="cuda", logger=tb_logger) # Use 1 GPU
    trainer.fit(model, dm)

    trainer.test(model, dm)

    # Test one image of the test set and plot the result
    x,y = next(iter(dm.test_dataloader()))
    out = model.forward(x)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x[0].permute(1, 2, 0)) # permute dimensions from (3, 128, 128) to (128, 128, 3)
    ax[0].set_title("Original Image")
    ax[1].imshow(out[0].permute(1, 2, 0))
    ax[1].set_title("Reconstructed Image")
    plt.show()


    # # Define a dummy input tensor
    # dummy_input = torch.randn(1, 3, 128, 128)
    # reconstructed_image = model(dummy_input)
    # print("Input shape:", dummy_input.shape)  # Expected: torch.Size([1, 3, 128, 128])
    # print("Reconstructed image shape:", reconstructed_image.shape)  # Expected: torch.Size([1, 3, 128, 128])
