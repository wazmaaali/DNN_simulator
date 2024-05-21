import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        
        image = Image.open(img_path).convert("RGB")
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                class_label, x_center, y_center, width, height = map(float, line.split())
                boxes.append([class_label, x_center, y_center, width, height])
        
        sample = {'image': image, 'boxes': boxes}
        if self.transform:
            sample = self.transform(sample)

        return sample
