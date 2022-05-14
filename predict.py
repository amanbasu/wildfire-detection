import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import CustomDataGenerator
from models import trans_unet, swin_unet, unet

BATCH_SIZE = 64
IMG_SIZE = 224                                                                  # segmenter default size
CHANNELS = 3
PRED_PATH = './'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

def numpify(array, filename, suffix):
    global PRED_PATH
    filename = PRED_PATH + filename.split('.')[0] + '_' + suffix
    np.save(filename, array)

def save_images(images, masks, preds, filename):
    images = images.numpy()
    masks = masks.numpy()
    preds = torch.sigmoid(preds).numpy()
    for i in range(len(images)):
        numpify(images[i], filename[i], 'input')
        numpify(masks[i], filename[i], 'mask')
        numpify(preds[i], filename[i], 'pred')

def predict(model):
    global BATCH_SIZE, SAVE_PATH, device

    testDataset = CustomDataGenerator(image_file='images_test',
                                    mask_file='masks_test', 
                                    root_dir='dataset',
                                    transform=transforms.Compose([
                                        Rescale(256),
                                        CenterCrop(IMG_SIZE),
                                        ToTensor(),
                                        ])
                                    )

    testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)
    testLoader = iter(testLoader)

    model.eval()
    with torch.no_grad():
        try:
            for batch in tqdm(testLoader):
                images = batch['image'].to(device)
                labels = batch['mask'].to(device)
                outputs = model(images)
                save_images(images.detach().cpu(), 
                            labels.detach().cpu(),
                            outputs.detach().cpu(),
                            batch['name'])
        except StopIteration:
            pass
        
def get_model(name):
    global IMG_SIZE, BATCH_SIZE, CHANNELS, device

    model, path = None, None
    if name == 'swin':
        model = swin_unet(IMG_SIZE, BATCH_SIZE).to(device)
        path = './train_output/model_checkpoint_swinUnet.pt'
    elif name == 'trans':
        model = trans_unet(IMG_SIZE).to(device)
        path = './train_output/model_checkpoint_transUnet.pt'
    elif name == 'unet':
        model = unet(n_channels=CHANNELS).to(device)
        path = './train_output/model_checkpoint_unetours.pt'

    return model, path
    
if __name__ == '__main__':

    # plug-in your model here
    NAME = sys.argv[1]
    try:
        PRED_PATH = sys.argv[2]
    except:
        pass
    
    model, SAVE_PATH = get_model(NAME)
    print(SAVE_PATH)

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    predict(model)
