import matplotlib.pyplot as plt
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
from image_inpainting.adversarial_discriminator import AdversarialDiscriminator
from image_inpainting.adversarial_generator import AdversarialGenerator
from image_inpainting.encoder import Encoder
from image_inpainting.decoder import Decoder
from image_inpainting.loss import JointLoss, ReconstructionLoss, AdversarialLoss
from image_inpainting.tiny_image_net_data_module import TinyImageNetDataModule
from image_inpainting.image_net_data_module import ImageNetDataModule


# Based on https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
class ContextEncoder(pl.LightningModule):
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000, reconstruction_loss_weight = 0.999, adversarial_loss_weight = 0.001):
        super(ContextEncoder, self).__init__()

        self.automatic_optimization = False # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.joint_loss = JointLoss(reconstruction_loss_weight=reconstruction_loss_weight, adversarial_loss_weight=adversarial_loss_weight)
        self.rec_loss = ReconstructionLoss()
        self.adv_loss = AdversarialLoss()
        self.test_psnr = 0

        self.generator = AdversarialGenerator(input_size=input_size, hidden_size=hidden_size)
        self.discriminator = AdversarialDiscriminator(input_channels=input_size[0])


    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        # They use a higher learning rate for context encoder (10 times) to that of adversarial discriminator (page 6 of the paper)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        return [generator_optimizer, discriminator_optimizer], [] # [] because there is no scheduler

    def training_step(self, batch, batch_idx):
        masked_imgs, true_masked_regions = batch

        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)

        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Generator
        self.toggle_optimizer(generator_optimizer) # we don't want to update the discriminator here
        joint_loss_val = self.joint_loss(
            context_encoder_outputs=generator_outputs,
            true_masked_regions=true_masked_regions,
            discriminator_fake_predictions=self.discriminator(generator_outputs),
            discriminator_real_predictions=self.discriminator(true_masked_regions)
        )

        self.manual_backward(joint_loss_val)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)

        # Discriminator
        self.toggle_optimizer(discriminator_optimizer)  # we don't want to update the generator here

        # these three steps were not really well explained in the paper, but it's a common practice to do it
        # and this is also described here: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
        discriminator_loss_val = self.adv_loss(fake_predictions=self.discriminator(generator_outputs.detach()), real_predictions=self.discriminator(true_masked_regions))
        self.manual_backward(discriminator_loss_val)
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        self.log('train_psnr', psnr_val, prog_bar=True)
        self.log('train_loss', joint_loss_val, prog_bar=True)
        return joint_loss_val

    def validation_step(self, batch, batch_idx):
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        loss_val = self.joint_loss(
            context_encoder_outputs=generator_outputs,
            true_masked_regions=true_masked_regions,
            discriminator_fake_predictions=self.discriminator(generator_outputs),
            discriminator_real_predictions=self.discriminator(true_masked_regions)
        )

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_loss', loss_val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        loss_val = self.joint_loss(
            context_encoder_outputs=generator_outputs,
            true_masked_regions=true_masked_regions,
            discriminator_fake_predictions=self.discriminator(generator_outputs),
            discriminator_real_predictions=self.discriminator(true_masked_regions)
        )

        self.test_psnr += psnr_val

        self.log('test_psnr', psnr_val, prog_bar=True)
        self.log('test_loss', loss_val, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_psnr = 0

    def on_test_epoch_end(self):
        self.test_psnr /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
        self.log('Final PSNR', self.test_psnr)



from pathlib import Path
import os

# Test the ContextEncoder
if __name__ == "__main__":
    data_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, "data", "imagenet")
    
    # Create an instance of the ContextEncoder
    # model = ContextEncoder(input_size=(3, 128, 128), hidden_size=4000)
    model = ContextEncoder.load_from_checkpoint("Context_Encoder_Inpainting/lightning_logs/version_1/checkpoints/epoch=6-step=516488.ckpt")
    # Training
    # dm = TinyImageNetDataModule(batch_size_train=64, batch_size_val=64, batch_size_test=64, num_workers=10, pin_memory=True, persistent_workers=True) # They recommend using the number of cores and set to True for GPUs https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    dm = ImageNetDataModule(data_dir=data_dir, batch_size_train=32, batch_size_val=32, batch_size_test=32, num_workers=10, pin_memory=True, persistent_workers=True) # They recommend using the number of cores and set to True for GPUs https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    tb_logger = pl_loggers.TensorBoardLogger("Context_Encoder_Inpainting")
    trainer = pl.Trainer(max_epochs=30, devices=-1, accelerator="cuda", logger=tb_logger)
    trainer.fit(model, dm)
    
    trainer.test(model, dm)

    # Test some images of the test set and plot the results

    # Load it
    model = ContextEncoder.load_from_checkpoint("Context_Encoder_Inpainting/lightning_logs/version_1/checkpoints/epoch=6-step=516488.ckpt")
    model.to("cuda")

    dm.prepare_data()
    dm.setup("test")

    x, y = next(iter(dm.test_dataloader()))
    x = x.to("cuda")
    out = model.forward(x)

    fig, ax = plt.subplots(5, 2, figsize=(10, 20))
    for i in range(min(5, x.shape[0])):
        original_img = x[i].cpu().permute(1, 2, 0).clone()
        true_masked_part = y[i].cpu().permute(1, 2, 0)

        original_img[32:96, 32:96, :] = true_masked_part
        original_img = original_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])

        ax[i, 0].imshow(original_img)
        ax[i, 0].set_title("Original Image")

        reconstructed_masked_part = out[i].detach().cpu().permute(1, 2, 0)
        reconstructed_img = x[i].cpu().permute(1, 2, 0).clone()
        reconstructed_img[32:96, 32:96, :] = reconstructed_masked_part
        reconstructed_img = reconstructed_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])

        ax[i, 1].imshow(reconstructed_img)
        ax[i, 1].set_title("Reconstructed Image")
    plt.show()