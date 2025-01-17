import numpy as np
import matplotlib.pyplot as plt
import imageio
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def insert_image_center(image, center_region):
    """Inserts the center region into the image.
    
    Args:
        image(np.array): The image to insert the center region into.
        center_region(np.array): The center region to insert into the image.
        
    Returns:
        (np.array): The image with the center region inserted.
    """
    new_image = np.copy(image)

    half_mask_size = center_region.shape[0] // 2
    half_image_size = image.shape[0] // 2

    mask_start = half_image_size - half_mask_size
    mask_end = half_image_size + half_mask_size

    new_image[mask_start:mask_end, mask_start:mask_end, :] = center_region

    return new_image

def print_results_images(inputs, true_masked_parts, predicted_center_regions, title, inverse_transform_function):
    """Prints the input image, the true masked part and the predicted center region for the first 5 images in the batch.
    
    Args:
        inputs(torch.Tensor): The input images.
        true_masked_parts(torch.Tensor): The true masked parts of the images.
        predicted_center_regions(torch.Tensor): The predicted center regions of the images.
        title(str): The title of the plot.
        inverse_transform_function(callable): The function to inverse the transformation of the images (undo normalization).
    """
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
    
def extract_images_from_tensorboard_logs(event_files_paths, tag_name):
    """Extracts the images from the tensorboard logs.
    
    Args:
        event_files_paths(list): The paths to the event files.
        tag_name(str): The tag name of the images (e.g. validation/inpainted_image_1).
    
    Returns:
        (list): The images extracted from the tensorboard logs.
    """
    epoch = 0
    size_guidance = {
        'images': 0, # I didn't set it first so only 4 images were fetched. Here it's removing the limit on the number of images returned
    }
    
    if not isinstance(event_files_paths, list):
        event_files_paths = [event_files_paths]
            
    images = []
    for event_path in event_files_paths:
        accumulator = event_accumulator.EventAccumulator(event_path, size_guidance)
        accumulator.Reload()
        
        if tag_name not in accumulator.Tags()['images']:
            continue
        
        for event in accumulator.Images(tag_name):
            img = event.encoded_image_string
            img = imageio.imread(img)
            img = Image.fromarray(img.astype(np.uint8))
            
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            text = f"Epoch: {epoch}"
            
            epoch += 1
            
            _, height = img.size
            position = (10, height - 15)
            draw.text(position, text, font=font, fill="green")
            
            images.append(img)
        epoch -= 1 # because if we have multiple paths it's because we reloaded the model, so it will print the same image twice
        if event_path != event_files_paths[-1]:
            images.pop(-1)
        

    return images

def create_gif(images, output_file_path, fps=3, pause_duration=3):
    """Creates a gif from images and save it in given path.
    
    Args:
        images(list): The list of images to create the gif from.
        output_file_path(str): The path to save the gif to.
        fps(int): The frames per second of the gif.
        pause_duration(int): The duration of the pause at the end of the gif (in number of frames).
    """
    with imageio.get_writer(output_file_path, mode='I', fps=fps, loop=0) as writer:
        for image in images:
            writer.append_data(image)
            
        for _ in range(pause_duration):
            writer.append_data(images[-1])