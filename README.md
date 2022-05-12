# Leveraging Vision Transformers for Enhanced Wildfire Detection and Characterization

In this project, we use the active fire dataset from https://github.com/pereira-gha/activefire and try to improve over their results. We use two Vision Transformer networks: [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) and [TransUnet](https://github.com/Beckschen/TransUNet), and one CNN-based UNet network.

File details:
- [UNet.py](UNet.py): Contains the pytorch code for UNet model.
- [evaluate.py](evaluate.py): Takes in the model name and evaluates the saved checkpoint on 4 metrics: precision, recall, f-score, and IoU.
- [generator.py](generator.py): Data generator code.
- [models.py](models.py): Returns the instances of different models used in this work.
- [predict.py](predict.py): Saves the inference result from the a checkpoint file.
- [train.py](train.py): Code to train a model.
- [transform.py](transform.py): Image transforms for data augmentation.

Command to run the scripts:
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
