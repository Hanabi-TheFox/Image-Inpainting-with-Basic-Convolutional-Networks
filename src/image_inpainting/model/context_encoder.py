import pytorch_lightning as pl
import torch
import torchmetrics

from image_inpainting.model.adversarial_discriminator import AdversarialDiscriminator
from image_inpainting.model.adversarial_generator import AdversarialGenerator
from image_inpainting.loss import JointLoss, ReconstructionLoss, AdversarialLoss
from image_inpainting.utils import insert_image_center

import numpy as np

# Based on https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
class ContextEncoder(pl.LightningModule):
    """Context Encoder model based on the paper. It combines a generator and a discriminator to inpaint images. This class is a PyTorch Lightning Module (it implements the training, validation and test steps).
    
    Attributes:
        automatic_optimization (bool): Whether to use automatic optimization or not. It's set to False because we need to update the generator and discriminator separately as mentioned in lighning documentation.
        input_size (tuple): The size of the input images. Default is (3, 128, 128).
        hidden_size (int): The size of the hidden layer. Default is 4000 (in the paper).
        psnr_metric (torchmetrics.image.PeakSignalNoiseRatio): The PSNR metric.
        joint_loss (JointLoss): The joint loss function.
        rec_loss (ReconstructionLoss): The reconstruction loss function.
        adv_loss (AdversarialLoss): The adversarial loss function.
        test_psnr (float): The PSNR value for the test set (average across all the batches).
        generator (AdversarialGenerator): The generator model.
        discriminator (AdversarialDiscriminator): The discriminator model.
        save_image_per_epoch (bool): Whether to save the first inpainted image (from the validation set) per epoch or not. Default is False.
        lr_g (float): The learning rate for the generator. Default is 0.002 as mentioned in the cited paper. Because the paper says "We use a higher learning rate for context encoder (10 times) to that of adversarial discriminator"
        lr_d (float): The learning rate for the discriminator. Default is 0.0002.
    """
    def __init__(self, input_size=(3, 128, 128), hidden_size=4000, reconstruction_loss_weight = 0.999, adversarial_loss_weight = 0.001, save_image_per_epoch=False, lr_g=0.002, lr_d=0.0002):
        """Initializes the ContextEncoder class.
        
        Args:
            input_size (tuple): The size of the input images. Default is (3, 128, 128).
            hidden_size (int): The size of the hidden layer. Default is 4000 (in the paper).
            reconstruction_loss_weight (float): The weight of the reconstruction loss in the joint loss. Default is 0.999 (in the paper).
            adversarial_loss_weight (float): The weight of the adversarial loss in the joint loss. Default is 0.001 (in the paper).
            save_image_per_epoch (bool): Whether to save the first inpainted image (from the validation set) per epoch or not. Default is False.
            lr_g (float): The learning rate for the generator. Default is 1e-3. Because the paper says "We use a higher learning rate for context encoder (10 times) to that of adversarial discriminator"
            lr_d (float): The learning rate for the discriminator. Default is 1e-4.
        """
        super(ContextEncoder, self).__init__()

        self.automatic_optimization = False # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.joint_loss = JointLoss(reconstruction_loss_weight=reconstruction_loss_weight, adversarial_loss_weight=adversarial_loss_weight)
        self.rec_loss = ReconstructionLoss()
        self.adv_loss = AdversarialLoss()
        self.test_psnr = 0
        self.lr_g = lr_g
        self.lr_d = lr_d

        self.generator = AdversarialGenerator(input_size=input_size, hidden_size=hidden_size)
        self.discriminator = AdversarialDiscriminator(input_channels=input_size[0])
        
        self.save_image_per_epoch = save_image_per_epoch  
        
        self.save_hyperparameters()

    def enable_save_image_per_epoch(self):
        """Enables saving the first inpainted image (from the validation set) per epoch."""
        self.save_image_per_epoch = True
        
    def disable_save_image_per_epoch(self):
        """Disables saving the first inpainted image (from the validation set) per epoch."""
        self.save_image_per_epoch = False
    
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        return self.generator(x)

    def configure_optimizers(self):
        """Configures the optimizers for the generator and discriminator. Here typically Adam with a learning rate of 1e-3 as mentioned in the paper
        
        Returns:
            list: The list of optimizers for the generator and discriminator (2 in our case: generator and discriminator).
            list: The list of schedulers (empty list in this case).
        """
        # They use a higher learning rate for context encoder (10 times) to that of adversarial discriminator (page 6 of the paper)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.9)) # suggested in the cited paper https://arxiv.org/pdf/1511.06434
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.9)) # same as above

        return [generator_optimizer, discriminator_optimizer], [] # [] because there is no scheduler

    def training_step(self, batch, batch_idx):
        """Training step of the model. It consists of two parts: the discriminator and the generator. The discriminator is updated first and then the generator.
        
        Args:
            batch (tuple): The batch of images.
            batch_idx (int): The index of the batch.
            
        Returns:
            dict: The dictionary containing the losses of the generator and discriminator.
        """
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        generator_optimizer, discriminator_optimizer = self.optimizers()
        
        # Discriminator
        
        self.toggle_optimizer(discriminator_optimizer)  # we don't want to update the generator here

        # these three steps were not really well explained in the paper, but it's a common practice to do it
        # and this is also described here: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
        discriminator_loss_val = self.adv_loss(fake_predictions=self.discriminator(generator_outputs.detach()), real_predictions=self.discriminator(true_masked_regions))

        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss_val)
        discriminator_optimizer.step()
        
        self.untoggle_optimizer(discriminator_optimizer)

        # Generator
        
        generator_outputs = self.forward(masked_imgs)
        discriminator_outputs = self.discriminator(generator_outputs)
        joint_loss_val = self.joint_loss(generator_outputs, true_masked_regions, discriminator_outputs)
        
        generator_optimizer.zero_grad()
        self.manual_backward(joint_loss_val)
        generator_optimizer.step()
        
        self.untoggle_optimizer(generator_optimizer)
        

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        
        self.log('train_psnr', psnr_val, prog_bar=True)
        self.log('train_loss_context_encoder', joint_loss_val, prog_bar=True)
        self.log('train_loss_discriminator', discriminator_loss_val, prog_bar=True)
        
        return {'train_joint_loss': joint_loss_val, 'train_loss_discriminator': discriminator_loss_val}

    def validation_step(self, batch, batch_idx):
        """Validation step of the model.
        
        Args:
            batch (tuple): The batch of images.
            batch_idx (int): The index of the batch.
            
        Returns:
            float: The joint loss value.
        """
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        discriminator_outputs = self.discriminator(generator_outputs)
        joint_loss_val = self.joint_loss(generator_outputs, true_masked_regions, discriminator_outputs)

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_loss', joint_loss_val, prog_bar=True)
        return joint_loss_val

    def test_step(self, batch, batch_idx):
        """Test step of the model.
        
        Args:
            batch (tuple): The batch of images.
            batch_idx (int): The index of the batch.
        
        Returns:
            float: The joint loss value.
        """
        masked_imgs, true_masked_regions = batch
        generator_outputs = self.forward(masked_imgs)

        psnr_val = self.psnr_metric(generator_outputs, true_masked_regions)
        discriminator_outputs = self.discriminator(generator_outputs)
        joint_loss_val = self.joint_loss(generator_outputs, true_masked_regions, discriminator_outputs)

        self.test_psnr += psnr_val

        self.log('test_psnr', psnr_val, prog_bar=True)
        self.log('test_loss', joint_loss_val, prog_bar=True)
        return joint_loss_val

    def on_test_epoch_start(self):
        """Resets the test PSNR value."""
        self.test_psnr = 0

    def on_test_epoch_end(self):
        """Calculates the final PSNR value for the test set."""
        self.test_psnr /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
        self.log('Final PSNR', self.test_psnr)
        
    def on_validation_end(self):
        """Saves the first inpainted image (from the validation set) per epoch if the flag is enabled."""
        x, _ = next(iter(self.trainer.datamodule.val_dataloader()))
        
        # x is a batch with many images but we only want the output for 3 or less images (depend on batch size)
        n_samples = min(3, len(x))
        x = x[:n_samples].to(self.device)
        out = self.forward(x)        
        
        x = x.cpu().clone()
        out = out.detach().cpu()
        
        for i in range(n_samples):
            reconstructed_image = x[i]
            reconstructed_image = self.trainer.datamodule.inverse_transform(reconstructed_image)
            reconstructed_masked_part = self.trainer.datamodule.inverse_transform(out[i])
            reconstructed_image = insert_image_center(reconstructed_image, reconstructed_masked_part).astype(np.uint8)
            
            # move channels to 1st dimension (expected by the add_image function)
            reconstructed_image = torch.tensor(reconstructed_image).permute(2, 0, 1)
            
            if self.save_image_per_epoch: 
                self.logger.experiment.add_image(
                    tag=f"validation/inpainted_image_{i}",
                    img_tensor=reconstructed_image,
                    global_step=self.global_step,
                )