# This script calculates optical flow between consecutive frames in a folder of images,
# saving the magnitude of the flow as images in an output folder.
# To run it, ensure you have OpenCV installed and provide the correct paths for input images and output folder.

import cv2
import os
import numpy as np
import time

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def calculate_optical_flow(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = sorted(os.listdir(input_folder))
    total_time = 0
    num_frames = len(file_list)

    for i in range(num_frames):
        filename1 = file_list[i]
        if i < num_frames - 1:
            filename2 = file_list[i + 1]

            if filename1.endswith(".jpg") or filename1.endswith(".png") and \
               filename2.endswith(".jpg") or filename2.endswith(".png"):
                input_path1 = os.path.join(input_folder, filename1)
                input_path2 = os.path.join(input_folder, filename2)

                frame1 = cv2.imread(input_path1)
                frame2 = cv2.imread(input_path2)

                # Preprocess frames
                gray1 = preprocess_image(frame1)
                gray2 = preprocess_image(frame2)

                start_time = time.time()

                # Calculate optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                end_time = time.time()
                time_taken = end_time - start_time
                total_time += time_taken

                print(f"Frame {i + 1}: Time taken for flux calculation: {time_taken:.4f} seconds")

                # Convert optical flow to polar coordinates
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Normalize magnitude for visualization
                magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

                # Convert magnitude to 8-bit unsigned int
                magnitude = np.uint8(magnitude)

                # Save the optical flow image with the same name as input image
                output_path = os.path.join(output_folder, filename1)
                cv2.imwrite(output_path, magnitude)

                print(f"Processed: {input_path1} and {input_path2}")

    # Calculate and display the average time taken for flux calculation
    if num_frames > 1:
        average_time = total_time / (num_frames - 1)
        print(f"Average time taken for flux calculation per frame: {average_time:.4f} seconds")

if __name__ == "__main__":
    input_folder = r"Path to input folder"
    output_folder = r"Path to output folder"
    calculate_optical_flow(input_folder, output_folder)
