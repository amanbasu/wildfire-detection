import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import CustomDataGenerator
from models import trans_unet, swin_unet, unet

INIT_EPOCH = 0
EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 224                                                                  
CHANNELS = 3
LEARNING_RATE = 0.001
STEPS = 173

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
       return 1 - (num/den)

def get_dataloader():
    trainDataset = CustomDataGenerator(image_file='images_train',
                                    mask_file='masks_train', 
                                    root_dir='dataset',
                                    transform=transforms.Compose([
                                        Rescale(256),
                                        RandomFlip(0.5),
                                        RandomRotate(0.5),
                                        RandomErase(0.2),
                                        RandomShear(0.2),
                                        RandomCrop(IMG_SIZE),
                                        ToTensor(),
                                        ])
                                    )

    # no transforms here, just resize the image
    valDataset = CustomDataGenerator(image_file='images_val',
                                    mask_file='masks_val', 
                                    root_dir='dataset',
                                    transform=transforms.Compose([
                                        Rescale(256),
                                        CenterCrop(IMG_SIZE),
                                        ToTensor(),
                                        ])
                                    )

    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    return iter(trainLoader), iter(valLoader)

def train(model, criterion, opt, scheduler):
    
    global INIT_EPOCH, EPOCHS, BATCH_SIZE, SAVE_PATH, device
    
    model.train()

    for epoch in range(INIT_EPOCH, EPOCHS):
        trainLoader, valLoader = get_dataloader()
        # when dataloader runs out of batches, it throws an exception 
        try:
            for batch in tqdm(trainLoader):
                images = batch['image'].to(device)                                                  
                labels = batch['mask'].to(device)
                
                opt.zero_grad(set_to_none=True)                                 # clear gradients w.r.t. parameters

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()                                                 # getting gradients
                opt.step()                                                      # updating parameters
                scheduler.step()                                                # to change the learing rate
        except StopIteration:
            pass

        # get model performace on val set
        with torch.no_grad():
            accuracy = []
            try:
                for batch in tqdm(valLoader):
                    images = batch['image'].to(device)
                    labels = batch['mask'].to(device)

                    outputs = model(images)
                    ls = criterion(outputs, labels).item()
                    accuracy += [1 - ls]
            except StopIteration:
                pass
            
            print('Epoch: {}/{} - accuracy: {:.4f}'.format(epoch+1, EPOCHS, np.mean(accuracy)))

        # save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                        }, SAVE_PATH)
            print('checkpoint saved.')

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

    criterion = dice_loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                    max_lr=LEARNING_RATE*10,
                                                    steps_per_epoch=STEPS,
                                                    pct_start=0.15,
                                                    epochs=EPOCHS
                                                    )

    # start from last checkpoint
    if INIT_EPOCH > 0:
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        INIT_EPOCH = checkpoint['epoch']
        # loss = checkpoint['loss']

    train(model, criterion, opt, scheduler)
