import numpy as np
import matplotlib.pyplot as plt


def insert_image_center(image, center_region):
    new_image = np.copy(image)

    half_mask_size = center_region.shape[0] // 2
    half_image_size = image.shape[0] // 2

    mask_start = half_image_size - half_mask_size
    mask_end = half_image_size + half_mask_size

    new_image[mask_start:mask_end, mask_start:mask_end, :] = center_region

    return new_image

def print_results_images(inputs, true_masked_parts, predicted_center_regions, title, inverse_transform_function):
    fig, ax = plt.subplots(5, 3, figsize=(10, 20))
    true_masked_parts = true_masked_parts.cpu().clone()
    inputs = inputs.cpu().clone()
    predicted_center_regions = predicted_center_regions.detach().cpu()

    for i in range(min(5, inputs.shape[0])):
        input_img = inputs[i]
        true_masked_part = true_masked_parts[i]
        
        input_img = inverse_transform_function(input_img)
        true_masked_part = inverse_transform_function(true_masked_part)
        original_img = insert_image_center(input_img, true_masked_part)

        ax[i, 0].imshow(original_img)
        ax[i, 0].set_title("Original Image")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(input_img)
        ax[i, 1].set_title("Input Image")
        ax[i, 1].axis("off")
        
        reconstructed_masked_part = predicted_center_regions[i]
        reconstructed_masked_part = inverse_transform_function(reconstructed_masked_part)
        reconstructed_image = insert_image_center(original_img, reconstructed_masked_part)
        
        ax[i, 2].imshow(reconstructed_image)
        ax[i, 2].set_title("Reconstructed Image")
        ax[i, 2].axis("off")
    fig.suptitle(title)
    plt.show()