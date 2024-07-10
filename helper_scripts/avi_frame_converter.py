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
            # Resize frame to 1920x1080
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
    video_path = r"\\vm-mue-filer01\ips_videostorage\Video-Sammlungen\TUGraz\indoor-cam-132.avi"
    
    # Output folder to save frames
    output_folder = r"C:\Users\dgn\Desktop\Internal_test_original_frames\indoor-cam-132"
    
    # Frame name prefix
    frame_prefix = "indoor-cam-132"
    
    # Convert video to frames
    video_to_frames(video_path, output_folder, frame_prefix)
