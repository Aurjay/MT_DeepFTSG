import torch
import numpy as np
import glob
import os
import pandas as pd
import imageio
import torch.nn.functional as F
import time
import torch.hub
from datetime import timedelta
from data.dataLoader_test import DeepFTSG_2_TestDataLoader
from nets.DeepFTSG_2 import DeepFTSG

device = torch.device('cpu')
print(device)

modelName = 'DeepFTSG_2'

print('**************************')
print(modelName)
print('**************************')

# network input size
imgWidth = 480
imgHeight = 320

# pretrained se-resnet50 on ImageNet
se_resnet_hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)

se_resnet_base_layers = []

# traverse each element of hub model and add it to a list
for name, m in se_resnet_hub_model.named_children():
    se_resnet_base_layers.append(m)

numClass = 1
model = DeepFTSG(numClass, se_resnet_base_layers).to(device)

# load trained model
model.load_state_dict(torch.load(r'C:\Users\dgn\Desktop\DeepFTSG-main\DeepFTSG-main\src\models\DeepFTSG_2.pt', map_location=torch.device('cpu')))
model.eval()

# Define specific paths for the different folders
folderDataPath = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Private\Original_frames\SiemensGehen20m\*.jpg'
folderBgSubPath = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models\DeepFTSG\SiemensGehen20m\BGS\*.jpg'
folderFluxPath = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models\DeepFTSG\SiemensGehen20m\Flux\*.jpg'

# paths to the test image, BGS, and flux
folderData = sorted(glob.glob(folderDataPath))
folderBgSub = sorted(glob.glob(folderBgSubPath))
folderFlux = sorted(glob.glob(folderFluxPath))

# Check if any of the folders are empty
if not folderData:
    raise FileNotFoundError(f"No images found in {folderDataPath}")
if not folderBgSub:
    raise FileNotFoundError(f"No BGS images found in {folderBgSubPath}")
if not folderFlux:
    raise FileNotFoundError(f"No flux images found in {folderFluxPath}")

print(len(folderData))

startIndex = len(folderData) - len(folderBgSub)
print(startIndex)

folderData = folderData[startIndex:]
print(len(folderData))

print(folderData[0])
print(folderBgSub[0])
print(folderFlux[0])

print('***************************')

print(folderData[-1])
print(folderBgSub[-1])
print(folderFlux[-1])

testDataset = DeepFTSG_2_TestDataLoader(folderData, folderBgSub, folderFlux, img_size=(imgHeight, imgWidth))
print(len(testDataset))

testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False)

# set mask path
maskDir = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models\DeepFTSG\SiemensGehen20m\Output'
# create path if not exist
if not os.path.exists(maskDir):
    os.makedirs(maskDir)

for i, (inputs, inputs2) in enumerate(testLoader):

    inputs = inputs.to(device)
    inputs2 = inputs2.to(device)

    inputs = inputs.float()
    inputs2 = inputs2.float()

    # Predict
    pred = model(inputs, inputs2)

    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()

    outPred = pred[0].squeeze()
    outPredNorm = 255 * outPred
    outPredUint8 = outPredNorm.astype(np.uint8)

    # Thresholding
    threshold_value = 127
    outPredThresholded = (outPredUint8 > threshold_value).astype(np.uint8) * 255

    # get frame name from original frame and replace in with bin and extension of jpg to png
    fname = os.path.basename(folderData[i]).replace('in', 'bin').replace('jpg', 'png')

    print(os.path.join(maskDir, fname))

    imageio.imwrite(os.path.join(maskDir, fname), outPredThresholded)

finalTime = time.time() - startTime
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(finalTime))
print(msg)
