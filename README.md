# Image-Inpainting-with-Basic-Convolutional-Networks
Objective : Fill in missing regions in an image by generating plausible and coherent content with the rest of the image.

# Install

## CUDA

Make sure that CUDA is installed on your machine. 
You can check the version of CUDA with the following command: `nvcc --version`.
It will be needed to install the correct version of PyTorch in the following steps.

## Conda

In environment.yaml, change the line `- torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` according to your CUDA version ([here](https://pytorch.org/get-started/locally/)).

### Create environment

If you don't have an existing conda environment, create one with the following command:

`conda env create --file environment.yaml`

### Update existing environment

If you already have an existing conda environment, update it with the following command:

`conda env update --file environment --prune`

## Without Conda

You first need to install PyTorch. You can find the installation instructions [here](https://pytorch.org/get-started/locally/).

If you don't want to use conda, you can install the required packages with pip : `pip install -r requirements.txt`.

## Check installation

`python -c "import torch; print(torch.cuda.is_available())"`

If the output is `True`, the installation is successful. Otherwise, check that CUDA is installed correctly or that the correct version of PyTorch is installed.

# Dataset

(downloaded automatically)
Tiny Image Net : http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Usage

1. `conda activate pytorch_env`
2. `python test_lighning.py`


# Authors

- Ethan PINTO
- Robin MENEUST