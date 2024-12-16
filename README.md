# Image-Inpainting-with-Basic-Convolutional-Networks
Objective : Fill in missing regions in an image by generating plausible and coherent content with the rest of the image. 
This is an implementation of the paper: Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). Context Encoders: Feature Learning by Inpainting. https://arxiv.org/abs/1604.07379


# Install

## CUDA

Make sure that CUDA is installed on your machine. 
You can check the version of CUDA with the following command: `nvcc --version`.
It will be needed to install the correct version of PyTorch in the following steps.

## Requirements installation

### Conda

In environment.yaml, change the line `- torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` according to your CUDA version ([here](https://pytorch.org/get-started/locally/)).

#### Create environment

If you don't have an existing conda environment, create one with the following command:

`conda env create -f environment.yaml`

#### Update existing environment

If you already have an existing conda environment, update it with the following command:

`conda env update -f environment.yaml --prune`

#### Activate the environment

`conda activate image_inpainting`

### Without Conda

You first need to install PyTorch. You can find the installation instructions [here](https://pytorch.org/get-started/locally/).

If you don't want to use conda, you can install the required packages with pip : `pip install -r requirements.txt`.

## Install the package

Finally, you have to install the package with: `pip install -e .`

## Check installation

`python -c "import torch; print(torch.cuda.is_available())"`

If the output is `True`, the installation is successful. Otherwise, check that CUDA is installed correctly or that the correct version of PyTorch is installed.

# Dataset

## Tiny image net

(downloaded automatically)
Tiny Image Net : http://cs231n.stanford.edu/tiny-imagenet-200.zip

## Image net

The dataset object of Imagenet expects a folder (e.g. nammed "imagenet") with 3 folders inside: train, val, test. Each of those has a list of folders with images inside.
The ImageNet datasets that we have collected and preprocessed (64x64 and 128x128) aren't available for now, since it might be against the ImageNet policy to publish it.

The original dataset is ILSVRC 2012: https://huggingface.co/datasets/ILSVRC/imagenet-1k

# Usage

Please refer to the notebooks for examples.

# Authors

- Ethan PINTO
- Robin MENEUST