import cv2
import os

def frames_to_video(input_folder, output_video, fps):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort the files numerically if needed

    frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(os.path.join(input_folder, image_file))
        video_writer.write(frame)

    video_writer.release()

input_folder = r"C:\Users\dgn\Desktop\cdnet-test-videos\boulevard\input"
output_video = r"C:\Users\dgn\Desktop\cdnet-test-videos\boulevard\boulevard.avi"
fps = 30  # Frames per second

frames_to_video(input_folder, output_video, fps)
