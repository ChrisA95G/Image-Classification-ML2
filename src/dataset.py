import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        # Assuming your labels are in a column named 'labels'
        # and are already multi-hot encoded as a list or numpy array
        self.labels = self.df['labels'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image ID for the current index
        image_id = self.df.iloc[idx]['ID']
        image_path = f'path/to/images/{image_id}_' # Base path

        # Your image loading logic goes here
        red_image = Image.open(image_path + 'red.png').convert('L')
        green_image = Image.open(image_path + 'green.png').convert('L')
        blue_image = Image.open(image_path + 'blue.png').convert('L')
        yellow_image = Image.open(image_path + 'yellow.png').convert('L')

        # Your preprocessing logic goes here
        red_tensor = self.transform(red_image)
        green_tensor = self.transform(green_image)
        blue_tensor = self.transform(blue_image)
        yellow_tensor = self.transform(yellow_image)

        image_tensor = torch.cat([red_tensor, green_tensor, blue_tensor, yellow_tensor], dim=0)
        
        # You'll also need normalization here, maybe pass it in the constructor
        mean4 = [0.5, 0.5, 0.5, 0.5]
        std4 = [0.5, 0.5, 0.5, 0.5]
        normalize4 = transforms.Normalize(mean=mean4, std=std4)
        image_tensor = normalize4(image_tensor)
        
        # Get the label and convert it to a tensor
        label_vector = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image_tensor, label_vector