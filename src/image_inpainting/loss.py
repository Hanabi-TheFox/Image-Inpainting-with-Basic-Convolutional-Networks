import torch
from torch import nn

class JointLoss(nn.Module):
    """Joint loss function combining the reconstruction loss and the adversarial loss for the context encoder described in the paper.
    
    Attributes:
        rec_loss_weight(float): The weight of the reconstruction loss in the joint loss function.
        adv_loss_weight(float): The weight of the adversarial loss in the joint loss function.
        rec_loss(ReconstructionLoss): The reconstruction loss function.
        adv_loss(AdversarialLoss): The adversarial loss function.
    """
    
    def __init__(self, reconstruction_loss_weight=0.999, adversarial_loss_weight=0.001):
        """Initializes the joint loss function with the given weights for the reconstruction and adversarial losses.
        
        Args:
            reconstruction_loss_weight: The weight of the reconstruction loss in the joint loss function.
            adversarial_loss_weight: The weight of the adversarial loss in the joint loss function.
        """
        super(JointLoss, self).__init__()
        self.rec_loss_weight = reconstruction_loss_weight
        self.adv_loss_weight = adversarial_loss_weight
        self.rec_loss = ReconstructionLoss()
        self.adv_loss = AdversarialLoss()

    def forward(self, context_encoder_outputs, true_masked_regions, discriminator_outputs):
        """Computes the joint loss function for the context encoder given the predictions, true image and discriminator outputs.
        
        Args:
            context_encoder_outputs(torch.Tensor): The output of the context encoder, i.e. the generated image for the masked part.
            true_masked_regions(torch.Tensor): The true masked regions of the input image.
            discriminator_outputs(torch.Tensor): The output of the discriminator for the generated image for the masked part.
        
        Returns:
            (torch.Tensor): The joint loss value for the context encoder.
        """
        
        
        # Here the paper doesn't really explain how to calculate the loss, and seems to
        # consider using also the discriminator loss for true image for the generator which is strange, especially since in their lua version of the paper they don't use it (they use fake only)
        # because it would imply that it's also trying to fool the discriminator on the true image, which is not the case
        # it would get penalized if the discriminator is bad at recognizing a true image (which is not the goal)
        
        rec_loss_val = self.rec_loss(context_encoder_outputs, true_masked_regions) # generator loss
        adv_loss_val = nn.BCELoss()(discriminator_outputs, torch.ones_like(discriminator_outputs)) # discriminator loss

        return self.rec_loss_weight * rec_loss_val + self.adv_loss_weight * adv_loss_val

class ReconstructionLoss(nn.Module):
    """Reconstruction loss function for the context encoder described in the paper.
    
    Attributes:
        overlapping_width_px(int): The width of the overlapping region in pixels. Tt's set to 7 in the paper.
        overlapping_weight_factor(int): The weight factor for the overlapping region in the loss function. It's set to 10 in the paper, i.e. overlapping regions (i.e. the borders of the predictions) are 10 times more important than the non-overlapping region.
    """
    
    def __init__(self, overlapping_width_px=7, overlapping_weight_factor=10):
        """Initializes the reconstruction loss function with the given parameters.
        
        Args:
            overlapping_width_px(int): The width of the overlapping region in pixels.
            overlapping_weight_factor(int): The weight factor for the overlapping region in the loss function. It's set to 10 in the paper, i.e. overlapping regions (i.e. the borders of the predictions) are 10 times more important than the non-overlapping region.
        """
        super(ReconstructionLoss, self).__init__()
        self.overlapping_width_px = overlapping_width_px
        self.overlapping_weight_factor = overlapping_weight_factor

    def forward(self, context_encoder_outputs, true_masked_regions):
        """Computes the reconstruction loss for the context encoder given the predictions and true image.
        
        Args:
            context_encoder_outputs(torch.Tensor): The output of the context encoder, i.e. the generated image for the masked part.
            true_masked_regions(torch.Tensor): The true masked regions of the input image.
            
        Returns:
            (torch.Tensor): The reconstruction loss value for the context encoder.
        """
        
        # compared to the paper:
        # - context_encoder_outputs is mask * F((1-M) * x): the generated inner part of the image
        # - true_masked_regions is mask * true_masked_regions is masked : the real inner part of the image

        # normalized distance L2
        
        # Here overlap is the region in the 7px border, and non-overlap is region inside

        mask_overlap = torch.ones_like(true_masked_regions)
        # dim 1 is batch, dim 2 is channels, dim 3 is height, dim 4 is width
        # so we only select height and width in [self.overlapping_width_px, max-self.overlapping_width_px]
        mask_overlap[:, :, self.overlapping_width_px:-self.overlapping_width_px, self.overlapping_width_px:-self.overlapping_width_px] = 0
        mask_non_overlap = 1 - mask_overlap

        diff = context_encoder_outputs - true_masked_regions
        overlap_diff = diff * mask_overlap
        non_overlap_diff = diff * mask_non_overlap
                
        # We use mse instead of l2, as they do in the lua version of the paper
        l2_cubed_overlap_mse = (overlap_diff ** 2).mean(dim=(1, 2, 3)) # not on the batch dimension
        l2_cubed_non_overlap_mse = (non_overlap_diff ** 2).mean(dim=(1, 2, 3)) # not on the batch dimension
        
        # Mean over batch
        loss = l2_cubed_overlap_mse * self.overlapping_weight_factor + l2_cubed_non_overlap_mse
        
        return loss.mean() # mean over batch

        # # We use MSE instead of the initial loss of the paper
        # return nn.MSELoss()(context_encoder_outputs, true_masked_regions)


class AdversarialLoss(nn.Module):
    """Adversarial loss function for the context encoder described in the paper.
    
    Attributes:
        loss_function(nn.BCELoss): The binary cross-entropy loss function used for the adversarial loss.
    """
    def __init__(self):
        """Initializes the adversarial loss function with the binary cross-entropy loss function."""
        super(AdversarialLoss, self).__init__()
        self.loss_function = nn.BCELoss()

    def forward(self, fake_predictions, real_predictions):
        """Computes the adversarial loss for the context encoder given the discriminator predictions for the generated and true images.
        
        Args:
            fake_predictions(torch.Tensor): The discriminator predictions for the generated image.
            real_predictions(torch.Tensor): The discriminator predictions for the true image.
            
        Returns:
            (torch.Tensor): The adversarial loss value for the context encoder.
        """
        # correction = 1e-8 # to avoid log(0) which is -inf (not explained in the paper but needed)
        # we add squeeze(-1) to remove the last dimension (e.g. (32, 1) -> (32))

        # how bad it is at labelling the true image as real (log(1) = 0 is what we want, log(0) -> -inf is bad)
        # log_d_real = torch.log(real_predictions + correction).mean()

        # how bad is it at labelling the generated image as fake (same, and here we invert the label of fake_predictions)
        # log_d_fake = torch.log(1 - fake_predictions + correction).mean()

        # the "E" in the formula in the paper seems to be the mean (E(X))
        # Here the model is trained to maximize the logistic likelihood, but we need to return a loss to be minimized, so we just take the opposite of the given formula
        # loss_val = -((log_d_real + log_d_fake) / 2)
        # return loss_val
                
        # We use BCE instead here, even though it's maybe different from the paper
        # Even the LUA version of the paper uses BCE
        
        # Actually BCE is the same as the formula in the paper because we have:
        
        # BCE_real = -y_real * log(real_predictions) - (1 - y_real) * log(1 - real_predictions)
        # but y_real = 1, so we have -log(real_predictions)
        
        # and BCE_fake = -y_fake * log(fake_predictions) - (1 - y_fake) * log(1 - fake_predictions)
        # but y_fake = 0, so we have -log(1 - fake_predictions) = -log(fake_predictions)
        
        fake_loss = self.loss_function(fake_predictions, torch.zeros_like(fake_predictions))
        real_loss = self.loss_function(real_predictions, torch.ones_like(real_predictions))
        return (fake_loss + real_loss) / 2
