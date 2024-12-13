import numpy as np

def insert_image_center(image, center_region):
    new_image = np.copy(image)

    half_mask_size = center_region.shape[0] // 2
    half_image_size = image.shape[0] // 2

    mask_start = half_image_size - half_mask_size
    mask_end = half_image_size + half_mask_size

    new_image[mask_start:mask_end, mask_start:mask_end, :] = center_region

    return new_image