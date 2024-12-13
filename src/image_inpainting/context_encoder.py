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
from image_inpainting.utils import insert_image_center

import numpy as np

# Based on https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
class ContextEncoder(pl.LightningModule):
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000, reconstruction_loss_weight = 0.999, adversarial_loss_weight = 0.001, save_image_per_epoch=False):
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
        
        self.save_image_per_epoch = save_image_per_epoch  

    def enable_save_image_per_epoch(self):
        self.save_image_per_epoch = True
        
    def disable_save_image_per_epoch(self):
        self.save_image_per_epoch = False
    
    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        # They use a higher learning rate for context encoder (10 times) to that of adversarial discriminator (page 6 of the paper)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3) # 1e-4 in the paper is for overlapping region (please refer to the losses in this package)

        return [generator_optimizer, discriminator_optimizer], [] # [] because there is no scheduler

    def training_step(self, batch, batch_idx):
        masked_imgs, true_masked_regions = batch

        generator_outputs = self.forward(masked_imgs)

        generator_optimizer, discriminator_optimizer = self.optimizers()
        
        # Discriminator
        
        self.toggle_optimizer(discriminator_optimizer)  # we don't want to update the generator here

        # these three steps were not really well explained in the paper, but it's a common practice to do it
        # and this is also described here: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
        discriminator_loss_val = self.adv_loss(fake_predictions=self.discriminator(generator_outputs.detach()), real_predictions=self.discriminator(true_masked_regions))
        self.manual_backward(discriminator_loss_val)
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        # Generator
        self.toggle_optimizer(generator_optimizer) # we don't want to update the discriminator here
        
        discriminator_outputs = self.discriminator(generator_outputs)
        joint_loss_val = self.joint_loss(generator_outputs, true_masked_regions, discriminator_outputs)
        
        self.manual_backward(joint_loss_val)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)
        

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        
        self.log('train_psnr', psnr_val, prog_bar=True)
        self.log('train_loss', joint_loss_val, prog_bar=True)
        
        return joint_loss_val

    def validation_step(self, batch, batch_idx):
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        discriminator_outputs = self.discriminator(generator_outputs)
        loss_val = self.joint_loss(generator_outputs, true_masked_regions, discriminator_outputs)

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_loss', loss_val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        discriminator_outputs = self.discriminator(generator_outputs)
        loss_val = self.joint_loss(generator_outputs, true_masked_regions, discriminator_outputs)

        self.test_psnr += psnr_val

        self.log('test_psnr', psnr_val, prog_bar=True)
        self.log('test_loss', loss_val, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_psnr = 0

    def on_test_epoch_end(self):
        self.test_psnr /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
        self.log('Final PSNR', self.test_psnr)
        
    def on_validation_end(self):
        x, _ = next(iter(self.trainer.datamodule.val_dataloader()))
        
        # x is a batch of images I only want tthe output for one image
        x = x[0].unsqueeze(0).to(self.device)
        out = self.forward(x)        
        
        reconstructed_image = x[0].cpu().clone()
        reconstructed_image = self.trainer.datamodule.inverse_transform(reconstructed_image)
        reconstructed_masked_part = dm.inverse_transform(out.detach().cpu()[0])
        reconstructed_image = insert_image_center(reconstructed_image, reconstructed_masked_part).astype(np.uint8)
        
        # move channels to 1st dimension
        reconstructed_image = torch.tensor(reconstructed_image).permute(2, 0, 1)
                
        if self.save_image_per_epoch: 
            self.logger.experiment.add_image(
                tag=f"validation/inpainted_image",
                img_tensor=reconstructed_image,
                global_step=self.global_step,
            )



from pathlib import Path
import os

# Test the ContextEncoder
if __name__ == "__main__":
    data_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")
    
    # Create an instance of the ContextEncoder
    model = ContextEncoder(input_size=(3, 128, 128), hidden_size=4000, save_image_per_epoch=True)
    
    # Or load it
    # model = ContextEncoder.load_from_checkpoint("Context_Encoder_Inpainting/lightning_logs/version_8/checkpoints/epoch=58-step=184434.ckpt")
    # model.enable_save_image_per_epoch()
    # model.to("cuda")
    
    # Training
    # They recommend using the number of cores and set to True for GPUs https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    dm = TinyImageNetDataModule(
        data_dir=os.path.join(data_dir, "tiny-imagenet-200"),
        batch_size_train=64, 
        batch_size_val=64, 
        batch_size_test=64, 
        num_workers=10,
        pin_memory=True, 
        persistent_workers=True
    )
    
    # dm = ImageNetDataModule(
    #     data_dir=os.path.join(data_dir, "imagenet"), 
    #     batch_size_train=32,
    #     batch_size_val=32,
    #     batch_size_test=32,
    #     num_workers=10, 
    #     pin_memory=True, 
    #     persistent_workers=True
    # )
    
    tb_logger = pl_loggers.TensorBoardLogger("Context_Encoder_Inpainting")
    trainer = pl.Trainer(max_epochs=30, devices=-1, accelerator="cuda", logger=tb_logger)
    
    trainer.fit(model, dm)
    
    trainer.test(model, dm)

    # Test some images of the test set and plot the results

    # Load it
    # model = ContextEncoder.load_from_checkpoint("Context_Encoder_Inpainting/lightning_logs/version_8/checkpoints/epoch=58-step=184434.ckpt")
    # model.enable_save_image_per_epoch()
    # model.to("cuda")

    dm.prepare_data()
    dm.setup("test")

    x, y = next(iter(dm.test_dataloader()))
    x = x.to("cuda")
    out = model.forward(x)

    fig, ax = plt.subplots(5, 2, figsize=(10, 20))
    for i in range(min(5, x.shape[0])):
        original_img = x[i].cpu().clone()
        true_masked_part = y[i].cpu().clone()
        
        original_img = dm.inverse_transform(original_img)
        true_masked_part = dm.inverse_transform(true_masked_part)
        original_img = insert_image_center(original_img, true_masked_part)

        ax[i, 0].imshow(original_img)
        ax[i, 0].set_title("Original Image")

        reconstructed_masked_part = out[i].detach().cpu()
        reconstructed_masked_part = dm.inverse_transform(reconstructed_masked_part)
        reconstructed_image = insert_image_center(original_img, reconstructed_masked_part)
        
        ax[i, 1].imshow(reconstructed_image)
        ax[i, 1].set_title("Reconstructed Image")
    plt.show()