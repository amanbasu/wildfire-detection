import torch
import numpy as np
import pandas as pd
import rasterio

BASE_PATH = '<your-base-path>'
IMAGES_PATH = BASE_PATH + 'dataset/images/patches/'
MASKS_PATH = BASE_PATH + 'dataset/masks/voting/'

class CustomDataGenerator(torch.utils.data.Dataset):

    def __init__(self, image_file, mask_file, root_dir, transform=None):
        # read file names from csv files
        self.image = pd.read_csv(f'{root_dir}/{image_file}.csv')
        self.masks = pd.read_csv(f'{root_dir}/{mask_file}.csv')
        
        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image['images'].iloc[idx]
        img = rasterio.open(IMAGES_PATH + img_path).read((7,6,2))               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE
        
        mask_path = self.masks['masks'].iloc[idx]
        # correct file names - add '_voting_'
        mask_path = mask_path.split('_')
        mask_path = '_'.join(mask_path[:-1]) + '_voting_' + mask_path[-1]
        mask = rasterio.open(MASKS_PATH + mask_path).read().transpose((1, 2, 0))
        mask = np.float32(mask > 0.5)                                           # apparently they are not thresholded
        
        sample = {'image': img, 'mask': mask, 'name': img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample
