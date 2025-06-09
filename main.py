from torch.utils.data import DataLoader
from src.dataset import ProteinDataset
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import timm
import torch.nn as nn
import torchmetrics # For calculating metrics

# --- Configuration ---
TRAIN_CSV_PATH = 'data/train.csv'
TRAIN_IMG_DIR = 'data/train/'
NUM_CLASSES = 28
BATCH_SIZE = 32 # Keep this as is for full runs, or reduce for very quick memory checks
NUM_EPOCHS = 10
DEBUG_MODE = True # Set to False for full training

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

    if DEBUG_MODE:
        print("--- RUNNING IN DEBUG MODE ---")
        # Use a small fraction of the data for quick testing
        train_df = train_df.sample(frac=0.02, random_state=42) # e.g., 1% of training data
        val_df = val_df.sample(frac=0.02, random_state=42)     # e.g., 1% of validation data

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

    # 1. Load the pre-trained model as usual
    model = timm.create_model(
        'vit_base_patch16_384',
        pretrained=True,
        num_classes=28
    )

    # 2. Get the weights from the original patch embedding layer (the first Conv2d)
    original_patch_embed = model.patch_embed.proj
    original_weights = original_patch_embed.weight.data # Shape: [768, 3, 16, 16]

    # 3. Create a new Conv2d layer with 4 input channels
    new_patch_embed = nn.Conv2d(
        in_channels=4,
        out_channels=original_patch_embed.out_channels,
        kernel_size=original_patch_embed.kernel_size,
        stride=original_patch_embed.stride,
        padding=original_patch_embed.padding
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # 4. Initialize the new weights by averaging the old ones and repeating
    # This transfers the learned feature knowledge.
    with torch.no_grad():
        # Average the weights across the 3 input channels
        new_weights = original_weights.mean(dim=1, keepdim=True)
        # Repeat this average for our 4 new input channels
        new_weights = new_weights.repeat(1, 4, 1, 1)
        new_patch_embed.weight.data = new_weights
        # Also copy over the bias if it exists
        if new_patch_embed.bias is not None:
            new_patch_embed.bias.data = original_patch_embed.bias.data

    # 5. Replace the model's original patch embedding layer with our new one
    model.patch_embed.proj = new_patch_embed

    # --- Training Loop ---
    current_epochs = NUM_EPOCHS
    if DEBUG_MODE:
        current_epochs = 3 # Train for only a couple of epochs in debug mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- Initialize Metrics ---
    # Using 'macro' average for F1 score as a common choice for multi-label
    # For debug mode, metrics on tiny datasets might not be very meaningful but ensure code runs
    train_f1_score = torchmetrics.F1Score(task="multilabel", num_labels=NUM_CLASSES, average='macro').to(device)
    train_hamming_dist = torchmetrics.HammingDistance(task="multilabel", num_labels=NUM_CLASSES).to(device)
    train_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=NUM_CLASSES).to(device) # Exact match ratio

    val_f1_score = torchmetrics.F1Score(task="multilabel", num_labels=NUM_CLASSES, average='macro').to(device)
    val_hamming_dist = torchmetrics.HammingDistance(task="multilabel", num_labels=NUM_CLASSES).to(device)
    val_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=NUM_CLASSES).to(device) # Exact match ratio
    for epoch in range(current_epochs):
        model.train() # Set the model to training mode
        total_train_loss = 0

        # Reset training metrics at the start of each epoch
        train_f1_score.reset()
        train_hamming_dist.reset()
        train_accuracy.reset()

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to the GPU if available
            images = images.to(device)
            # Labels for loss should be float, for metrics typically int
            labels_float = labels.to(device).float()
            labels_int = labels.to(device).int()

            # 1. Forward pass
            outputs = model(images)

            # 2. Calculate loss
            loss = loss_fn(outputs, labels_float)

            # 3. Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate predictions for metrics
            preds = (torch.sigmoid(outputs) > 0.5).int()
            train_f1_score.update(preds, labels_int)
            train_hamming_dist.update(preds, labels_int)
            train_accuracy.update(preds, labels_int)

            if batch_idx > 0 and (batch_idx % 10 == 0 or DEBUG_MODE): # Log more frequently in debug mode
                print(f"Epoch {epoch+1}/{current_epochs}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        epoch_train_f1 = train_f1_score.compute()
        epoch_train_hamming = train_hamming_dist.compute()
        epoch_train_acc = train_accuracy.compute()
        print(f"Epoch {epoch+1}/{current_epochs}, Avg Training Loss: {avg_train_loss:.4f}, Training F1: {epoch_train_f1:.4f}, Training Hamming: {epoch_train_hamming:.4f}, Training Accuracy: {epoch_train_acc:.4f}")

        # --- Validation Loop ---
        model.eval() # Set the model to evaluation mode
        total_val_loss = 0
        val_f1_score.reset()
        val_hamming_dist.reset()
        val_accuracy.reset()
        printed_sample_comparison = False # Flag to print only one sample comparison per epoch

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels_float = labels.to(device).float()
                labels_int = labels.to(device).int()

                outputs = model(images)
                loss = loss_fn(outputs, labels_float)
                total_val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_f1_score.update(preds, labels_int)
                val_hamming_dist.update(preds, labels_int)
                val_accuracy.update(preds, labels_int)

                # Print a sample comparison in DEBUG_MODE for the first batch of the epoch
                if DEBUG_MODE and not printed_sample_comparison and len(labels_int) > 0:
                    print(f"\n--- Sample Prediction vs. True Label (Epoch {epoch+1}) ---")
                    print(f"Predicted: {preds[0].cpu().numpy()}")
                    print(f"True     : {labels_int[0].cpu().numpy()}")
                    printed_sample_comparison = True

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_val_f1 = val_f1_score.compute()
        epoch_val_hamming = val_hamming_dist.compute()
        epoch_val_acc = val_accuracy.compute()
        print(f"Epoch {epoch+1}/{current_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation F1: {epoch_val_f1:.4f}, Validation Hamming: {epoch_val_hamming:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

if __name__ == "__main__":
    main()
