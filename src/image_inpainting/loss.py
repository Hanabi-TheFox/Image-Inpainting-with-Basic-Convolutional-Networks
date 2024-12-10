from torch import nn, log
from torch.linalg import matrix_norm

class JointLoss(nn.Module):
    def __init__(self, reconstruction_loss_weight=0.999, adversarial_loss_weight=0.001):
        super(JointLoss, self).__init__()
        self.rec_loss_weight = reconstruction_loss_weight
        self.adv_loss_weight = adversarial_loss_weight
        self.rec_loss = ReconstructionLoss()
        self.adv_loss = AdversarialLoss()

    def forward(self, predictions, targets):
        rec_loss_val = self.rec_loss(predictions, targets)
        adv_loss_val = self.adv_loss(predictions, targets)
        return self.rec_loss_weight * rec_loss_val + self.classification_loss_weight * adv_loss_val

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, encoder_out_masked_x, masks, inputs):
        # the element wise product in the paper is the Hadamard product
        # F is the encoder model

        # encoder_out_masked_x is F(Hadamard_prod((1-mask), input))
        # val = Hadamard product of mask and prediction
        val = masks * encoder_out_masked_x # TODO not sure if this is correct, check again
        # normalized distance L2
        return matrix_norm(val - inputs, ord=2) # TODO or MSE ?

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, decoder_out_dx, decoder_out_masked_dfx):
        # decoder_out_dx is the output of the decoder model for x (D(x))
        # decoder_out_masked_dfx is the output of the decoder model for the output of the encoder (D(F(Hadamard_prod((1-mask), input))))
        log_out_x = log(decoder_out_dx)
        log_out_masked_x = log(1 - decoder_out_masked_dfx)

        return log_out_x + log_out_masked_x # TODO check if this is correct, this is not exactly the same as how it's in the paper