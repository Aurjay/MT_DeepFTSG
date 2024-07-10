import cv2
import os
import time

def perform_background_subtraction(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Process each image in the input folder
    total_time = 0
    num_images = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = cv2.imread(input_path)

            # Start timer
            start_time = time.time()

            # Apply background subtraction
            fg_mask = bg_subtractor.apply(image)

            # End timer
            end_time = time.time()
            time_taken = end_time - start_time
            total_time += time_taken
            num_images += 1

            # Save the foreground mask
            cv2.imwrite(output_path, fg_mask)

            print(f"Processed: {input_path}. Time taken: {time_taken:.4f} seconds")

    # Calculate and display the average time taken for background subtraction
    if num_images > 0:
        average_time = total_time / num_images
        print(f"Average time taken for background subtraction per image: {average_time:.4f} seconds")

if __name__ == "__main__":
    input_folder = r"I:\Werkstudenten\Deepak_Raj\DATASETS\Private\Original_frames\SiemensGehen20m"
    output_folder = r"I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models\DeepFTSG\SiemensGehen20m\BGS"
    perform_background_subtraction(input_folder, output_folder)
