import torch
import numpy as np
import glob
import os
import imageio
import torch.nn.functional as F
import time
import torch.hub
import cv2
from datetime import timedelta
from torchvision import transforms
from nets.DeepFTSG_2 import DeepFTSG
import re

# Device configuration
device = torch.device('cpu')
print(device)

# Model parameters
model_name = 'DeepFTSG_2'
img_width, img_height = 480, 320

print('**************************')
print(model_name)
print('**************************')

# Load pretrained se-resnet50 on ImageNet
se_resnet_hub_model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=True)

# Extract base layers from the pretrained model
se_resnet_base_layers = list(se_resnet_hub_model.children())

# Initialize the DeepFTSG model
num_class = 1
model = DeepFTSG(num_class, se_resnet_base_layers).to(device)
model.load_state_dict(torch.load(r'C:\Users\dgn\Desktop\DeepFTSG-main\DeepFTSG-main\src\models\DeepFTSG_2.pt', map_location=device))
model.eval()

# Define paths
folder_data_path = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Public\Public\cars-illumination\*.jpg'
mask_dir = r'C:\Users\dgn\Desktop\DeepFTSG-main\DeepFTSG-main\test_output\cars-illumination'
os.makedirs(mask_dir, exist_ok=True)

# Natural sort function
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# Load image paths and sort them naturally
folder_data = sorted(glob.glob(folder_data_path), key=natural_sort_key)
if not folder_data:
    raise FileNotFoundError(f"No images found in {folder_data_path}")

print(len(folder_data))

# Define transformations
resize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
])

# Background subtraction and optical flow calculation
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

def perform_background_subtraction(image, bg_subtractor):
    return bg_subtractor.apply(image)

def preprocess_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculate_optical_flow(frame1, frame2):
    gray1, gray2 = preprocess_image(frame1), preprocess_image(frame2)
    try:
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    except AttributeError:
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(gray1, gray2, None)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude)

def postprocess_mask(mask):
    # Morphological operations for noise reduction
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Preprocessing and inference
start_time = time.time()

for i in range(len(folder_data) - 1):
    img_path1, img_path2 = folder_data[i], folder_data[i + 1]
    frame1, frame2 = cv2.imread(img_path1), cv2.imread(img_path2)

    # Perform background subtraction
    bg_mask = perform_background_subtraction(frame1, bg_subtractor)
    bg_mask = postprocess_mask(bg_mask)

    # Calculate optical flow
    flux_image = calculate_optical_flow(frame1, frame2)
    flux_image = postprocess_mask(flux_image)

    # Preprocess frames for model input
    frame1_resized = resize_transform(frame1).unsqueeze(0).float().to(device)
    bg_mask_resized = resize_transform(bg_mask).unsqueeze(0).float().to(device)
    flux_image_resized = resize_transform(flux_image).unsqueeze(0).float().to(device)
    inputs2 = torch.cat([bg_mask_resized, flux_image_resized, flux_image_resized], dim=1)

    # Predict
    pred = model(frame1_resized, inputs2)

    # Post-processing prediction
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()[0].squeeze()
    out_pred_uint8 = (pred * 255).astype(np.uint8)
    threshold_value = 127
    out_pred_thresholded = (out_pred_uint8 > threshold_value).astype(np.uint8) * 255
    out_pred_thresholded = postprocess_mask(out_pred_thresholded)
    
    # Save the output image
    new_fname = f'output-img{i+1:04}.jpg'
    imageio.imwrite(os.path.join(mask_dir, new_fname), out_pred_thresholded)

# Final performance metrics
end_time = time.time()
total_time = end_time - start_time
num_frames = len(folder_data) - 1
fps = num_frames / total_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(total_time))
print(msg)
print(fps)
