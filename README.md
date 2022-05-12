# Leveraging Vision Transformers for Enhanced Wildfire Detection and Characterization

<img src="https://img.shields.io/github/stars/amanbasu/wildfire-detection?color=0088ff"/> <img src="https://img.shields.io/github/forks/amanbasu/wildfire-detection?color=ff8800"/> <img src="https://img.shields.io/badge/torch-1.9.0+cu111-green?logo=pytorch"/> <img src="https://img.shields.io/badge/python-3.9.6-blue?logo=python"/>

In this project, we use the active fire dataset from https://github.com/pereira-gha/activefire ([data link](https://drive.google.com/drive/folders/1GIcAev09Ye4hXsSu0Cjo5a6BfL3DpSBm)) and try to improve over their results. We use two Vision Transformer networks: [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) and [TransUnet](https://github.com/Beckschen/TransUNet), and one CNN-based UNet network. We show that ViT can outperform well-trained and specialized CNNs to detect wildfires on a previously published dataset of LandSat-8 imagery (Pereira et al.). One of our ViTs outperforms the baseline CNN comparison by 0.92%. However, we find our own implementation of CNN-based UNet to perform best in every category, showing their sustained utility in image tasks. Overall, ViTs are comparably capable in detecting wildfires as CNNs, though well-tuned CNNs are still the best technique for detecting wildfire with our UNet providing an IoU of 93.58%, better than the baseline UNet by some 4.58%. 

## File description

- [UNet.py](UNet.py): Contains the pytorch code for UNet model.
- [evaluate.py](evaluate.py): Takes in the model name and evaluates the saved checkpoint on 4 metrics: precision, recall, f-score, and IoU.
- [generator.py](generator.py): Data generator code.
- [models.py](models.py): Returns the instances of different models used in this work.
- [predict.py](predict.py): Saves the inference result from the a checkpoint file.
- [train.py](train.py): Code to train a model.
- [transform.py](transform.py): Image transforms for data augmentation.

## Commands

```python
# Train
python train.py <model-name>
## Example
python train.py unet

# Evaluate
python evaluate.py <model-name>
## Example
python evaluate.py unet

# Save predictions
python predict.py <model-name> <image-path>
## Example
python predict.py unet predictions/unet/
```

## Results

 Method | Precision | Recall | F-score | IoU
 :----- | :-------: | :----: | :-----: | :--:
 U-Net (10c) | 92.90 | 95.50 | **94.20** | 89.00
 U-Net (3c) | 91.90 | 95.30 | 93.60 | 87.90
 U-Net-Light (3c) | 90.20 | **96.50** | 93.20 | 87.30
 TransUNet | 88.46 | 86.88 | 87.66 | 87.49
 Swin-Unet | 88.28 | 92.30 | 90.24 | 89.93
 Our UNet | **93.37** | 93.96 | 93.67 | **93.58**
