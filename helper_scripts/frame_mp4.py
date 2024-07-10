# This script reads frames from a directory and writes them to an MP4 video file.
# Make sure you have OpenCV installed and provide the correct paths for input frames
# and the output video file path.

import cv2
import os

# Directory containing the frames
frames_dir = r'Path\to\input\folder'

# Output video file path
output_video_path = 'original.mp4'

# Get the list of frame filenames
frame_files = sorted(os.listdir(frames_dir))

# Define video properties based on the first frame
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, _ = first_frame.shape
fps = 30  # You can adjust this based on your frame rate

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Loop through frames and write to video
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    out.write(frame)

# Release video writer
out.release()

print(f"Video saved to: {output_video_path}")
