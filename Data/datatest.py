import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)  # Reads the image as a PyTorch tensor (C, H, W)
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        
        if self.transform:
            image = self.transform(image)
        return image, label



dataset = CustomImageDataset(annotations_file='data/label.csv', img_dir='data/images', transform=None)
