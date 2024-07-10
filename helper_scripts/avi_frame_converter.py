# This script extracts frames from a given video file and saves them as JPG images in a specified output folder.
# To run it, ensure you have OpenCV installed and provide the correct paths for the video file and output folder.

import cv2
import os

def video_to_frames(video_path, output_folder, frame_prefix):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of digits needed for frame numbering
    num_digits = len(str(total_frames))

    # Loop through each frame
    for frame_num in range(total_frames):
        # Grab frame without decoding
        cap.grab()

        # Decode and retrieve the grabbed frame
        ret, frame = cap.retrieve()
        
        # Check if frame is retrieved successfully
        if ret:
            # Resize frame to 380x244
            frame = cv2.resize(frame, (380, 244))

            # Save frame as JPG image with custom naming convention
            frame_name = f"{frame_prefix}.{frame_num:0{num_digits}d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"Processed frame {frame_num}/{total_frames}")
        else:
            print(f"Error retrieving frame {frame_num}/{total_frames}")

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Input video file path
    video_path = r"Path to video file"
    
    # Output folder to save frames
    output_folder = r"Path to output folder"
    
    # Frame name prefix
    frame_prefix = "Prefix for frame names"
    
    # Convert video to frames
    video_to_frames(video_path, output_folder, frame_prefix)
