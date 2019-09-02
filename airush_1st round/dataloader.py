import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nsml import DATASET_PATH


def cutout(*, mask_size=24, cutout_inside=False, mask_color=(255, 255, 255)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def train_dataloader(input_size=128,
                     batch_size=64,
                     num_workers=0,
                     ):

    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images')
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)

    #this line contains data augmentation
    dataloader = DataLoader(
        AIRushDataset(image_dir, train_meta_data, label_path=train_label_path,
                      transform=transforms.Compose([transforms.RandomGrayscale(p=0.2),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomChoice([
                                                        transforms.Compose([
                                                            transforms.Resize((input_size + 30, input_size + 60)),
                                                            transforms.RandomCrop(input_size)
                                                        ]),
                                                        transforms.Compose([
                                                            transforms.Resize((input_size + 60, input_size + 60)),
                                                            transforms.RandomCrop(input_size)
                                                        ]),
                                                        transforms.Compose([
                                                            transforms.Resize((input_size, input_size)),
                                                            cutout()
                                                        ]),
                                                        transforms.Resize((input_size, input_size))
                                                    ]),
                                                    # transforms.RandomRotation(30),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ])
                      ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader


class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform

        if self.label_path is not None:
            self.label_matrix = np.load(label_path)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(self.meta_data['package_id'].iloc[idx]), str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load()  # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)

        if self.label_path is not None:
            tags = torch.tensor(np.argmax(self.label_matrix[idx]))  # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img
