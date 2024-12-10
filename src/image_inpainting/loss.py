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

    def forward(self, context_encoder_outputs, true_masked_regions, discriminator_fake_predictions, discriminator_real_predictions):
        rec_loss_val = self.rec_loss(context_encoder_outputs, true_masked_regions) # generator loss
        adv_loss_val = self.adv_loss(discriminator_fake_predictions, discriminator_real_predictions) # discriminator loss

        return self.rec_loss_weight * rec_loss_val + self.adv_loss_weight * adv_loss_val

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, context_encoder_outputs, true_masked_regions):
        # compared to the paper:
        # - context_encoder_outputs is mask * F((1-M) * x): the generated inner part of the image
        # - true_masked_regions is mask * true_masked_regions is masked : the real inner part of the image

        # normalized distance L2
        diff = context_encoder_outputs - true_masked_regions
        l2_cubed = torch.norm(diff, p=2, dim=(1, 2, 3)) ** 2
        return l2_cubed.mean() # mean over the batch


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, fake_predictions, real_predictions):
        # we add squeeze(-1) to remove the last dimension (e.g. (32, 1) -> (32))

        # how bad it is at labelling the true image as real (log(1) = 0 is what we want, log(0) -> -inf is bad)
        log_d_real = log(real_predictions).squeeze(-1)

        # how bad is it at labelling the generated image as fake (same, and here we invert the label of fake_predictions)
        log_d_fake = log(1 - fake_predictions).squeeze(-1)

        # the "E" in the formula in the paper seems to be the mean (E(X))
        # Here the model is trained to maximize the logistic likelihood, but we need to return a loss to be minimized, so we just take the opposite of the given formula
        loss_val = -((log_d_real + log_d_fake) / 2)
        return loss_val.mean() # mean over the batch
