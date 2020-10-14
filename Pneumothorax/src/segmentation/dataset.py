import os 
import glob
import torch 
import numpy as np
import pandas as pd 
from PIL import Image,ImageFile
from tqdm  import tqdm
from collections  import defaultdict
from torchvision import transforms
from pathlib import Path
from albumentations import(
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma, 
    ShiftScaleRotate
)

ImageFile.LOAD_TRUNCATED_IAMGES = True

class SIIMDataset(torch.utils.data.Dataset):

    def __init__(self,image_ids,transform=True,preprocessing_fn = None ):

        self.data=defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        TRAIN_PATH = Path('../../input/png_images')
        MASK_PATH = Path('../../input/png_masks')
        self.aug = Compose(
            [
            ShiftScaleRotate(
                shift_limit= 0.0625,
                scale_limit= 0.1,
                rotate_limit = 10,
                p=0.8
            ),
            OneOf(
                [
                RandomGamma(
                    gamma_limit=(90,110)
                ),
                RandomBrightnessContrast(
                    brightness_limit= 0.1,
                    contrast_limit = 0.1
                ),

            ],
            p = 0.5,
            ),
            ]
        )

        for i in range(len(image_ids)):
            files = glob.glob(os.path.join(TRAIN_PATH, image_ids[i],"*.png"))
            self.data[i]= {
                "img_path": os.path.join(
                    TRAIN_PATH, image_ids[i]
                ),
                "mask_path" : os.path.join(
                    MASK_PATH,image_ids[i] 
                ),
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        img_path = self.data[item]['img_path']
        mask_path = self.data[item]['mask_path']

        img = Image.open(img_path)
        img = img.convert("RGB")

        img = np.array(img)

        mask = Image.open(mask_path)
        mask =np.array(mask)
        #print(mask)

        mask = (mask >=1 ).astype("float32")
        

        if self.transform is True:
            augmented = self.aug(image = img, mask = mask)
            img = augmented['image']
            mask =augmented['mask']

        img = self.preprocessing_fn(img)

        return {
            "image" : transforms.ToTensor()(img),
            "mask":transforms.ToTensor()(mask).float()
        }

        


