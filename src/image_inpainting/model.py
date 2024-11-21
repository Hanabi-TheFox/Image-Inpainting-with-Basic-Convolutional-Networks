import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
from image_inpainting.loss import JointLoss

class ContextEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ContextEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.loss_function = JointLoss()

        # TODO : Define the model here


    def forward(self,x):
        raise Exception("Not implemented")

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

        self.log('train_acc', psnr)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        loss_function = nn.CrossEntropyLoss()

        outputs = self.forward(x)
        loss = loss_function(outputs, y)
        psnr = self.psnr_metric(outputs, y)

        self.log('val_acc', psnr)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x,y = batch
        loss_function = nn.CrossEntropyLoss()

        outputs = self.forward(x)
        loss = loss_function(outputs, y)

        self.psnr += torchmetrics.functional.accuracy(outputs, y, task="multiclass", num_classes=self.num_classes)

        self.log('test_loss', loss)
        self.log('test_psnr', self.psnr / (batch_idx+1))

    def on_test_epoch_start(self):
        self.psnr = 0

    def on_test_epoch_end(self):
        self.psnr /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
        self.log('Final PSNR', self.psnr)