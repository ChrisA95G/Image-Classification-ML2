from torch.utils.data import DataLoader
from src.dataset import ProteinDataset
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
# --- Configuration ---
TRAIN_CSV_PATH = 'data/train.csv'
TRAIN_IMG_DIR = 'data/train/'
NUM_CLASSES = 28
BATCH_SIZE = 32

    # Function to convert the 'Target' string to a multi-hot encoded vector
def create_multi_hot_label(target_string):
        labels = [int(i) for i in target_string.split(' ')]
        multi_hot_vector = torch.zeros(NUM_CLASSES)
        multi_hot_vector[labels] = 1
        return multi_hot_vector

def main():

    # --- 1. Load and Preprocess the CSV ---

    # Load the csv file
    df = pd.read_csv(TRAIN_CSV_PATH)


    # Apply this function to create a new column with the correct label format
    df['multi_hot_labels'] = df['Target'].apply(create_multi_hot_label)

    # --- 2. Split Data into Training and Validation Sets ---

    # We'll split the dataframe, for example, an 80/20 split.
    # stratify=df['Target'] can help ensure label distribution is similar in both sets, which is good practice.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    # --- 4. Create the DataLoaders ---

    # Create dataset instances
    train_dataset = ProteinDataset(df=train_df, image_dir=TRAIN_IMG_DIR)
    val_dataset = ProteinDataset(df=val_df, image_dir=TRAIN_IMG_DIR)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Example of how to get one batch ---
    print("\nExample batch:")
    images, labels = next(iter(train_loader))
    print(f"Images batch shape: {images.shape}") # Should be [32, 4, 384, 384]
    print(f"Labels batch shape: {labels.shape}") # Should be [32, 28]


if __name__ == "__main__":
    main()
