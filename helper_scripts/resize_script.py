import os
from PIL import Image

def resize_images(image_folder, output_folder, target_width=380, target_height=244):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image files in the input folder
    image_files = os.listdir(image_folder)

    # Filter out only PNG and JPG files
    image_files = [file for file in image_files if file.lower().endswith(('.png','.jpg'))]

    # Sort the files to ensure they are processed in the same order
    image_files.sort()

    resized_paths = []

    for file in image_files:
        # Load the image
        image_path = os.path.join(image_folder, file)
        image = Image.open(image_path)

        # Resize the image to the target dimensions
        resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Construct the output path
        output_path = os.path.join(output_folder, file)

        # Save the resized image to the output folder
        resized_image.save(output_path)

        # Append the path of the resized image to the list
        resized_paths.append(output_path)

    return resized_paths

# Example usage:
input_folder = r'C:\Users\dgn\Desktop\CVATVideoSegmentationsByJaved\badmintonSegmentation'
output_folder = r'C:\Users\dgn\Desktop\CVATVideoSegmentationsByJaved\badmintonSegmentation_resized'
resized_paths = resize_images(input_folder, output_folder)
print("Resized images saved to:", resized_paths)
