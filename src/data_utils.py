import torch
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from src.config import TRAIN_CSV_PATH, NUM_CLASSES, DEBUG_MODE, POS_WEIGHT

def create_multi_hot_label(target_string: str) -> torch.Tensor:
    """Converts a space-separated string of target labels into a multi-hot tensor."""
    labels = [int(i) for i in target_string.split(" ")]
    multi_hot_vector = torch.zeros(NUM_CLASSES)
    multi_hot_vector[labels] = 1
    return multi_hot_vector


def prepare_data():
    """Loads, preprocesses, and splits the training data.
    Also calculates pos_weight for handling class imbalance.
    """
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["multi_hot_labels"] = df["Target"].apply(create_multi_hot_label) # type: ignore

    # For MultilabelStratifiedShuffleSplit, labels need to be a NumPy array
    X = df["Id"].values # Or any other feature/identifier column
    y = np.stack(df["multi_hot_labels"].values) #type: ignore

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in msss.split(X, y):
        train_df = df.iloc[train_index].copy() 
        val_df = df.iloc[val_index].copy()

    if DEBUG_MODE:
        train_df = train_df.sample(frac=0.1, random_state=42)
        val_df = val_df.sample(frac=0.1, random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    if not train_df.empty:
        all_labels_train = torch.stack(train_df["multi_hot_labels"].tolist())
        class_counts_train = all_labels_train.sum(dim=0)
        num_train_samples = len(train_df)
        pos_weight = (num_train_samples - class_counts_train) / (class_counts_train + 1e-6) # Add epsilon to avoid division by zero
        pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)
        if POS_WEIGHT:
            print(f"Calculated pos_weight for loss function: {pos_weight}")
    else:
        pos_weight = torch.ones(NUM_CLASSES)

    return train_df, val_df, pos_weight