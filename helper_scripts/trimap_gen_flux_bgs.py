# This script generates a trimap using background subtraction and flux calculation images.
# It combines these two types of masks to create a trimap with foreground, background, and unknown regions.
# To run it, ensure you have OpenCV installed and provide the correct paths for background subtraction (BGS),
# flux (FLUX) images, and the output folder for trimaps.

import cv2
import os
import numpy as np

# Function to generate trimap using background subtraction and flux calculation
def generate_trimap(bg_img, flux_img, prev_mask):
    # Read background subtraction and flux images
    bg_mask = cv2.imread(bg_img, cv2.IMREAD_GRAYSCALE)
    flux_mask = cv2.imread(flux_img, cv2.IMREAD_GRAYSCALE)

    # Combine foreground mask from background subtraction and moving object mask
    combined_mask = cv2.bitwise_or(bg_mask, flux_mask)

    # Fill holes in the mask
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    # Combine with previous mask for temporal consistency
    if prev_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, prev_mask)

    # Create trimap
    trimap = np.zeros_like(combined_mask)
    trimap[combined_mask == 255] = 255
    trimap[(combined_mask == 0) & (bg_mask == 0)] = 128  # Unknown region

    return trimap, bg_mask

# Main function
def main():
    # Folder containing background subtraction (bgs) and flux images
    bgs_folder = r'Path to background subtraction images'
    flux_folder = r'Path to flux images'
    output_folder = r'Path to output folder'

    # Get list of background subtraction and flux images
    bgs_images = sorted(os.listdir(bgs_folder))
    flux_images = sorted(os.listdir(flux_folder))

    prev_mask = None

    for bg_img, flux_img in zip(bgs_images, flux_images):
        # Generate trimap
        trimap, prev_mask = generate_trimap(os.path.join(bgs_folder, bg_img), 
                                            os.path.join(flux_folder, flux_img), 
                                            prev_mask)

        # Save trimap
        output_path = os.path.join(output_folder, f"trimap_{bg_img}")
        cv2.imwrite(output_path, trimap)

    print("Trimaps saved successfully.")

if __name__ == "__main__":
    main()
