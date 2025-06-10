from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import ProteinDataset
from src.augmentations import CustomAugmentationTransform
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import timm
import torch.nn as nn
import time
import torchmetrics
from tqdm import tqdm

# Configuration
CUSTOM_DATA_PATH = Path("/Users/niklashohn/Desktop/human-protein-atlas-image-classification")
TRAIN_CSV_PATH = CUSTOM_DATA_PATH/"train.csv"
TRAIN_IMG_DIR = CUSTOM_DATA_PATH/"train/"
NUM_CLASSES = 28
NUM_EPOCHS = 50
BATCH_SIZE = 32
DEBUG_MODE = True
NUM_EPOCHS_DEBUG_MODE = 5
FREEZE_BACKBONE = True
MAX_GRAD_NORM = 1.0


def create_multi_hot_label(target_string):
    labels = [int(i) for i in target_string.split(" ")]
    multi_hot_vector = torch.zeros(NUM_CLASSES)
    multi_hot_vector[labels] = 1
    return multi_hot_vector


def prepare_data():
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["multi_hot_labels"] = df["Target"].apply(create_multi_hot_label)
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    if DEBUG_MODE:
        train_df = train_df.sample(frac=0.01, random_state=42)
        val_df = val_df.sample(frac=0.01, random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    return train_df, val_df


def create_datasets(train_df, val_df):
    train_transform = CustomAugmentationTransform(
        apply_augmentation=True,
        rotation_degrees=45,
        stretch_scale_range=(0.75, 1.25),
        stretch_shear_degrees=(-15, 15, -15, 15),
    )
    val_transform = CustomAugmentationTransform(apply_augmentation=False)

    train_dataset = ProteinDataset(
        df=train_df, image_dir=TRAIN_IMG_DIR, transform=train_transform
    )
    val_dataset = ProteinDataset(
        df=val_df, image_dir=TRAIN_IMG_DIR, transform=val_transform
    )

    return train_dataset, val_dataset


def create_model():
    model = timm.create_model(
        "vit_base_patch16_384",
        pretrained=True,
        num_classes=28,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )

    # Get original 3-channel weights
    original_patch_embed = model.patch_embed.proj
    original_weights = original_patch_embed.weight.data  # Shape: [768, 3, 16, 16]

    # Create new 4-channel projection
    new_patch_embed = nn.Conv2d(
        in_channels=4,
        out_channels=original_patch_embed.out_channels,
        kernel_size=original_patch_embed.kernel_size,
        stride=original_patch_embed.stride,
        padding=original_patch_embed.padding,
    )

    # Protein-specific channel initialization
    with torch.no_grad():
        # Initialize new weights tensor
        new_weights = torch.zeros_like(new_patch_embed.weight.data)

        # Assign channels based on biological role:
        new_weights[:, 0] = original_weights[:, 1]  # Green (protein of interest)
        new_weights[:, 1] = original_weights[:, 2]  # Blue (nucleus)
        new_weights[:, 2] = original_weights[:, 0]  # Red (microtubules)
        new_weights[:, 3] = 0.5 * (
            original_weights[:, 0] + original_weights[:, 1]
        )  # Yellow (ER)

        new_patch_embed.weight.data = new_weights

        if new_patch_embed.bias is not None:
            new_patch_embed.bias.data = original_patch_embed.bias.data.clone()

    model.patch_embed.proj = new_patch_embed
    return model


def setup_training(model):
    if FREEZE_BACKBONE:
        # Enhanced freezing strategy with patch_embed
        for name, param in model.named_parameters():
            param.requires_grad = any(
                [
                    name.startswith("head."),
                    name.startswith("norm."),
                    name.startswith("patch_embed."),  # Critical for protein channels
                    any(name.startswith(f"blocks.{i}.mlp.") for i in range(4, 8)),
                    any(name.startswith(f"blocks.{i}.norm") for i in range(4, 8)),
                    any(name.startswith(f"blocks.{i}.") for i in range(8, 12)),
                ]
            )

        # Differential learning rates
        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n.startswith("head.")
                ],
                "lr": 3e-4,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n.startswith("patch_embed.")
                ],
                "lr": 1e-4,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(n.startswith(f"blocks.{i}.") for i in range(8, 12))
                ],
                "lr": 1e-5,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(n.startswith(f"blocks.{i}.mlp.") for i in range(4, 8))
                    or any(n.startswith(f"blocks.{i}.norm") for i in range(4, 8))
                ],
                "lr": 5e-5,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    # Improved scheduler: Warmup + Cosine Annealing
    warmup_epochs = 5
    cosine_epochs = 45
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], [warmup_epochs]
    )

    return optimizer, scheduler


def setup_metrics(device):
    metrics = {
        "train": {
            "f1": torchmetrics.F1Score(
                task="multilabel", num_labels=NUM_CLASSES, average="macro"
            ).to(device),
            "hamming": torchmetrics.HammingDistance(
                task="multilabel", num_labels=NUM_CLASSES
            ).to(device),
            "exact_match": torchmetrics.ExactMatch(
                task="multilabel", num_labels=NUM_CLASSES
            ).to(device),
        },
        "val": {
            "f1": torchmetrics.F1Score(
                task="multilabel", num_labels=NUM_CLASSES, average="macro"
            ).to(device),
            "hamming": torchmetrics.HammingDistance(
                task="multilabel", num_labels=NUM_CLASSES
            ).to(device),
            "exact_match": torchmetrics.ExactMatch(
                task="multilabel", num_labels=NUM_CLASSES
            ).to(device),
        },
    }
    return metrics


def train_epoch(model, train_loader, optimizer, loss_fn, metrics, device):
    model.train()
    total_loss = 0
    for metric in metrics["train"].values():
        metric.reset()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for batch_idx, (images, labels) in pbar:
        images = images.to(device)
        labels_float = labels.to(device).float()
        labels_int = labels.to(device).int()

        outputs = model(images)
        loss = loss_fn(outputs, labels_float)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int()

        metrics["train"]["f1"].update(preds, labels_int)
        metrics["train"]["hamming"].update(preds, labels_int)
        metrics["train"]["exact_match"].update(preds, labels_int)

        if batch_idx > 0 and (batch_idx % 10 == 0 or DEBUG_MODE):
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, loss_fn, metrics, device, epoch):
    model.eval()
    total_loss = 0
    printed_sample = False

    for metric in metrics["val"].values():
        metric.reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, total=len(val_loader), desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels_float = labels.to(device).float()
            labels_int = labels.to(device).int()

            outputs = model(images)
            loss = loss_fn(outputs, labels_float)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            metrics["val"]["f1"].update(preds, labels_int)
            metrics["val"]["hamming"].update(preds, labels_int)
            metrics["val"]["exact_match"].update(preds, labels_int)

            if DEBUG_MODE and not printed_sample and len(labels_int) > 0:
                print("\nSample Prediction vs. True Label:")
                print(f"Predicted: {preds[0].cpu().numpy()}")
                print(f"True     : {labels_int[0].cpu().numpy()}")
                printed_sample = True

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(val_loader)


def main():
    train_df, val_df = prepare_data()
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    train_dataset, val_dataset = create_datasets(train_df, val_df)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = create_model()
    optimizer, scheduler = setup_training(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    metrics = setup_metrics(device)
    loss_fn = nn.BCEWithLogitsLoss()
    current_epochs = NUM_EPOCHS_DEBUG_MODE if DEBUG_MODE else NUM_EPOCHS

    start_time = time.time()
    for epoch in range(current_epochs):
        print(f"\nEpoch {epoch + 1}/{current_epochs}")
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, metrics, device
        )
        avg_val_loss = validate_epoch(
            model, val_loader, loss_fn, metrics, device, epoch
        )

        scheduler.step()

        print("\nMetrics:")
        print(f"Train - Loss: {avg_train_loss:.4f}, F1: {metrics['train']['f1'].compute():.4f}, "
              f"Hamming: {metrics['train']['hamming'].compute():.4f}, "
              f"Exact Match: {metrics['train']['exact_match'].compute():.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, F1: {metrics['val']['f1'].compute():.4f}, "
              f"Hamming: {metrics['val']['hamming'].compute():.4f}, "
              f"Exact Match: {metrics['val']['exact_match'].compute():.4f}")

    print(f"\nTotal training time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
