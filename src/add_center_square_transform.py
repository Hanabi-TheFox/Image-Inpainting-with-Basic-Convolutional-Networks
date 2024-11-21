import torch
import numpy as np
import cv2

class AddCenterSquareTransform:
    def __init__(self, output_size=128, square_fraction=0.25):
        """
        output_size : int
            Taille des images de sortie (carrées).
        square_fraction : float
            Fraction de la taille de l'image occupée par le carré blanc (ex: 0.25 = 1/4).
        """
        self.output_size = output_size
        self.square_fraction = square_fraction

    def __call__(self, image):
        # Redimensionner l'image à la taille de sortie
        image = cv2.resize(image, (self.output_size, self.output_size))
        
        # Calculer les dimensions du carré blanc
        square_size = int(self.output_size * self.square_fraction)
        start = (self.output_size - square_size) // 2
        end = start + square_size
        
        # Créer le masque binaire
        mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
        mask[start:end, start:end] = 1
        
        # Ajouter le carré blanc
        image[start:end, start:end] = 255  # Valeur maximale pour un carré blanc
        
        # Convertir l'image et le masque en tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normaliser entre 0 et 1
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        return image_tensor, mask_tensor