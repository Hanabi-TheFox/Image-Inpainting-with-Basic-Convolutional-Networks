import torch
from torch import nn, log
from torch.linalg import matrix_norm

class JointLoss(nn.Module):
    def __init__(self, reconstruction_loss_weight=0.999, adversarial_loss_weight=0.001):
        super(JointLoss, self).__init__()
        self.rec_loss_weight = reconstruction_loss_weight
        self.adv_loss_weight = adversarial_loss_weight
        self.rec_loss = ReconstructionLoss()
        self.adv_loss = AdversarialLoss()

    def forward(self, context_encoder_outputs, true_masked_regions, discriminator_outputs):
        # Here the paper doesn't really explain how to calculate the loss, and seems to
        # consider using also the discriminator loss for true image for the generator which is strange, especially since in their lua version of the paper they don't use it (they use fake only)
        # because it would imply that it's also trying to fool the discriminator on the true image, which is not the case
        # it would get penalized if the discriminator is bad at recognizing a true image (which is not the goal)
        
        rec_loss_val = self.rec_loss(context_encoder_outputs, true_masked_regions) # generator loss
        adv_loss_val = nn.BCELoss()(discriminator_outputs, torch.ones_like(discriminator_outputs)) # discriminator loss

        return self.rec_loss_weight * rec_loss_val + self.adv_loss_weight * adv_loss_val

class ReconstructionLoss(nn.Module):
    def __init__(self, overlapping_width_px=7, overlapping_weight_factor=10):
        super(ReconstructionLoss, self).__init__()
        self.overlapping_width_px = overlapping_width_px
        self.overlapping_weight_factor = overlapping_weight_factor

    def forward(self, context_encoder_outputs, true_masked_regions):
        # compared to the paper:
        # - context_encoder_outputs is mask * F((1-M) * x): the generated inner part of the image
        # - true_masked_regions is mask * true_masked_regions is masked : the real inner part of the image

        # normalized distance L2

        mask_overlap = torch.ones_like(true_masked_regions)
        # dim 1 is batch, dim 2 is channels, dim 3 is height, dim 4 is width
        # so we only select height and width in [self.overlapping_width_px, max-self.overlapping_width_px]
        mask_overlap[:, :, self.overlapping_width_px:-self.overlapping_width_px, self.overlapping_width_px:-self.overlapping_width_px] = 0
        mask_non_overlap = 1 - mask_overlap

        diff = context_encoder_outputs - true_masked_regions
        overlap_diff = diff * mask_overlap
        non_overlap_diff = diff * mask_non_overlap
        
        # l2_cubed_overlap_diff = (torch.norm(overlap_diff, p=2, dim=(1, 2, 3)) ** 2) # not on the batch dimension
        # l2_cubed_non_overlap_diff = (torch.norm(non_overlap_diff, p=2, dim=(1, 2, 3)) ** 2) # not on the batch dimension
        
        # We use mse instead of l2, as they do in the lua version of the paper
        l2_cubed_overlap_mse = (overlap_diff ** 2).mean(dim=(1, 2, 3))
        l2_cubed_non_overlap_mse = (non_overlap_diff ** 2).mean(dim=(1, 2, 3))
        
        # Mean over batch
        loss = l2_cubed_overlap_mse + self.overlapping_weight_factor * l2_cubed_non_overlap_mse
        
        return loss.mean() # mean over batch

        # # We use MSE instead of the initial loss of the paper
        # return nn.MSELoss()(context_encoder_outputs, true_masked_regions)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, fake_predictions, real_predictions):
        # we add squeeze(-1) to remove the last dimension (e.g. (32, 1) -> (32))

        # how bad it is at labelling the true image as real (log(1) = 0 is what we want, log(0) -> -inf is bad)
        # log_d_real = log(real_predictions).squeeze(-1)

        # how bad is it at labelling the generated image as fake (same, and here we invert the label of fake_predictions)
        # log_d_fake = log(1 - fake_predictions).squeeze(-1)

        # the "E" in the formula in the paper seems to be the mean (E(X))
        # Here the model is trained to maximize the logistic likelihood, but we need to return a loss to be minimized, so we just take the opposite of the given formula
        # loss_val = -((log_d_real + log_d_fake) / 2)

        # We use BCE instead here, even though it's maybe different from the paper
        # Even the LUA version of the paper uses BCE
        return (nn.BCELoss()(real_predictions, torch.ones_like(real_predictions)) + nn.BCELoss()(fake_predictions, torch.zeros_like(fake_predictions))) / 2
        # return loss_val.mean() # mean over the batch

# Test the encoder
if __name__ == "__main__":
    # Define a dummy input tensor (batch size = 1, channels = 3, height = 128, width = 128)
     # zeroes like
    dummy_x = torch.zeros(1, 3, 128, 128)
    dummy_y = torch.full((1, 3, 128, 128), 1)
    
    rec_loss = ReconstructionLoss()
    rec_loss_val = rec_loss(dummy_x, dummy_y)
    
    print(rec_loss_val)
