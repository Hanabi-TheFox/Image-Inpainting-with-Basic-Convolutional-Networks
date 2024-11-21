# Image-Inpainting-with-Basic-Convolutional-Networks
Objectif principal : Remplir des régions manquantes dans une image en générant un contenu plausible et cohérent avec le reste de l'image.

# Install

1. `conda create --name pytorch_env`
2. `conda activate pytorch_env`
3. `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia` (for Windows, see https://pytorch.org/get-started/locally/ for other versions or OS)
4. `conda install lightning -c conda-forge`


## Check

`python -c "import torch; print(torch.cuda.is_available())"`

# Use

1. `conda activate pytorch_env`
2. 