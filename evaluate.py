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
IMG_SIZE = 224                                                                  # lower image size for random crops (data augmentation)
CHANNELS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

class dice_loss(torch.nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.smooth = 1.

    def forward(self, logits, labels):
        logf = torch.sigmoid(logits).view(-1)
        labf = labels.view(-1)
        
        intersection = (logf * labf).sum()
        num = 2. * intersection + self.smooth
        den = logf.sum() + labf.sum() + self.smooth

        return num/den

def statistics(y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)

    return tn, fp, fn, tp

def show_stats(y_pred, y_true, accuracy):
    _, fp, fn, tp = statistics(np.array(y_true), np.array(y_pred))
    P = 100 * float(tp)/(tp + fp)
    R = 100 * float(tp)/(tp + fn)
    F = (2 * P * R)/(P + R)

    print(f'Precision: {P:.4f} - Recall: {R:.4f} - F-score: {F:.4f} - IoU: {100 * np.mean(accuracy):.4f}')

def evaluate(model, criterion):
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
    
    model.eval()
    with torch.no_grad():
        y_pred_all, y_true_all, accuracy = [], [], []
        iter = 0
        try:
            for batch in tqdm(testLoader):
                iter += 1
                images = batch['image'].to(device)
                labels = batch['mask'].to(device)

                outputs = model(images)
                accuracy += [criterion(outputs, labels).item()]

                y_pred = torch.sigmoid(outputs).contiguous().view(-1,).to('cpu').numpy()
                y_true = labels.contiguous().view(-1,).to('cpu').numpy()

                y_pred_all.append((y_pred))
                y_true_all.append(y_true)

                if iter % 5 == 0:
                    show_stats(y_pred_all, y_true_all, accuracy)

        except StopIteration:
            pass
    
        show_stats(y_pred_all, y_true_all, accuracy)

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
    model, SAVE_PATH = get_model(NAME)
    print(SAVE_PATH)
    
    # load weights
    criterion = dice_loss()

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate(model, criterion)
