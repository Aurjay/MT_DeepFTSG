# This script converts all PNG images in a specified folder to 24-bit RGB format, removing any alpha channels.
# To run it, ensure you have the Pillow library installed and provide the correct path to the folder containing the images.

from PIL import Image
import os

def convert_images_to_24bit(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only PNG files
    image_files = [file for file in files if file.lower().endswith('.png')]
    
    for file_name in image_files:
        # Construct the full path to the image file
        full_path = os.path.join(folder_path, file_name)
        
        # Open the image
        image = Image.open(full_path)
        
        # Convert to RGB (24-bit) if the image has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Save the converted image with the same name
        image.save(full_path)

# Example usage:
folder_path = r'Path to folder containing PNG images'
convert_images_to_24bit(folder_path)
