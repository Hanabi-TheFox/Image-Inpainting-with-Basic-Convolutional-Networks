# import pytorch_lightning as pl
# import torch
#
# class ContextEncoder(pl.LightningModule):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(ContextEncoder, self).__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#
#         # TODO : Define the model here
#
#
#     def forward(self,x):
#         raise Exception("Not implemented")
#
#     def configure_optimizers(self):
#         # TODO : Choose your optimizer : https://pytorch.org/docs/stable/optim.html
#         # Like in the 1st TP I've chosen here SGD, but other such as AdamW can also be used
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#         return optimizer
#
#     def training_step(self, batch, batch_idx):
#         # TODO : Define your Training Step
#         # This method is pretty much similar to what your did in the Tutorial to train your model.
#         x,y = batch
#         loss_function = nn.CrossEntropyLoss()
#
#         outputs = self.forward(x)
#         loss = loss_function(outputs, y)
#         acc = torchmetrics.functional.accuracy(outputs, y, task="multiclass", num_classes=self.num_classes) # https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html Note: At the beginning it seems that this task is a binary classification, but sincee we are classifying numbers in 0-10, it should by multiclass classification
#
#         # Backpropagation and learning -> already done with Lightning, we can still modify it for instance using the function backward (https://pytorch-lightning.readthedocs.io/en/0.10.0/introduction_guide.html)
#
#         # Don't remove the next line, you will understand why later
#         self.log('train_acc', acc)
#         self.log('train_loss', loss)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         # TODO : Define your Validation Step
#         # What is the difference between the Training and the Validation Step ?
#         # -> In the Validation step, we don't use the (Validation) data to train
#         # the model (no parameter update), we only evaluate the model after each
#         #step during training (no backpropagation)
#         x,y = batch
#         loss_function = nn.CrossEntropyLoss()
#
#         outputs = self.forward(x)
#         loss = loss_function(outputs, y)
#         acc = torchmetrics.functional.accuracy(outputs, y, task="multiclass", num_classes=self.num_classes)
#
#         # Don't remove the next line, you will understand why later
#         self.log('val_acc', acc)
#         self.log('val_loss', loss)
#
#     def test_step(self, batch, batch_idx):
#         # TODO : Define your Test Step
#         # What is the difference between the Training, Validation and Test Step ?
#
#         # -> Both the test set and validation set aren't use to train the model.
#         # The difference between the 2 is that the test set is only used once
#         # at the end, it should not be used to adjust our hyperparameters
#         # if needed. Typically, we don't use the test set during the evaluation
#         # step so that the model is not biased, it's not adjusted depending on the
#         # test set. It's like we ignore the test set until the end: until the Test step
#
#         # To summarize:
#         # - Training step : Adjust the model parameters (weights...) using the training set. Done for each epoch
#         # - Validation step : Evaluate the model on a different data set: the validation set. Done for each epoch after the training step
#         # - Test step : Evaluate the model on a different data set: the test set. Done only once at the end
#
#         x,y = batch
#         loss_function = nn.CrossEntropyLoss()
#
#         outputs = self.forward(x)
#         loss = loss_function(outputs, y)
#
#         # We accumulate every accuracy
#         self.acc += torchmetrics.functional.accuracy(outputs, y, task="multiclass", num_classes=self.num_classes)
#
#         # Don't remove the next line, you will understand why later
#         self.log('test_loss', loss)
#         # In the given code we don't divide by the current number of batches done, but self.acc is here a sum of accuracies so we have to do it to get the average current accuracy accuracy
#         # We can also use instead an intermediate var to keep the batch accuracy computed with the function accuracy() and just print it
#         self.log('test_acc', self.acc / (batch_idx+1))
#
#     def on_test_epoch_start(self):
#         self.acc = 0
#
#     def on_test_epoch_end(self):
#         self.acc /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
#         self.log('Final Accuracy', self.acc)