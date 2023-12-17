import os
import shutil
import random


def random_sample(original_path, output_path, sample_size):
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all files in the original directory
    all_files = os.listdir(original_path)

    # Randomly select a subset of files
    sampled_files = random.sample(all_files, sample_size)

    # Copy the sampled files to the output directory
    for file_name in sampled_files:
        file_path = os.path.join(original_path, file_name)
        shutil.copy(file_path, output_path)


# Example usage
original_dataset_path = r'mouse_images'
output_dataset_path = r'mouse_image_sample'
desired_sample_size = 275  # Adjust the size based on your needs

random_sample(original_dataset_path, output_dataset_path, desired_sample_size)
