import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ProteinDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir
        # Define transforms inside the dataset for clarity
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor()
        ])
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image ID and the pre-processed label tensor
        row = self.df.iloc[idx]
        image_id = row['Id']
        label_vector = row['multi_hot_labels']

        # Construct file paths for the four channels
        base_path = f"{self.image_dir}/{image_id}_"
        colors = ['red', 'green', 'blue', 'yellow']
        
        # Load, transform, and stack the images
        image_tensors = []
        for color in colors:
            image = Image.open(base_path + f"{color}.png").convert('L')
            image_tensors.append(self.transform(image))
            
        image_tensor = torch.cat(image_tensors, dim=0)
        
        # Apply normalization
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor, label_vector
