## Overview

This is an inference script for DeepFTSG implementation of the Master thesis 'Comparison of Different Anomaly-Detection-Algorithms and Neural Network Driven Background-Subtraction Models to Detect Unexpected Objects in Video Streams'
## Dependencies

Install all the packages in requirements.txt file. The code was executed in a Python 3.10.7 envirionment.

## Setup
```bash
pip install -r requirements.txt
```

# or
```bash
conda install --file requirements.txt
```


## Usage
Download the pretrained weights from [DeepFTSG GitHub repository](https://github.com/CIVA-Lab/DeepFTSG) and specify the appropriate folder in `src/Infer_DeepFTSG_2_all_in_one_final_script.py`.

To run the script:

```bash
python src/Infer_DeepFTSG_2_all_in_one_final_script.py
```

## Reference
This inference code has been refenced and modified from [DeepFTSG GitHub repository](https://github.com/CIVA-Lab/DeepFTSG)
