import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ProteinDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.custom_transform = transform
        self.colors = ["red", "green", "blue", "yellow"]

        self.base_transform = transforms.Compose(
            [transforms.Resize((384, 384)), transforms.ToTensor()]
        )

        self.normalize = transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_path = f"{self.image_dir}/{row['Id']}_"

        image_tensors = [
            self.base_transform(Image.open(f"{base_path}{color}.png").convert("L"))
            for color in self.colors
        ]

        image_tensor = torch.cat(image_tensors, dim=0)

        if self.custom_transform:
            image_tensor = self.custom_transform(image_tensor)

        return self.normalize(image_tensor), row["multi_hot_labels"]
