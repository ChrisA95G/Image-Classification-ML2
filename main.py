import time
import torch
import torch.nn as nn
from pathlib import Path
from src.dataset import ProteinDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.augmentations import CustomAugmentationTransform
from src.data_utils import prepare_data
from src.engine import train_epoch, validate_epoch
from src.submission import generate_submission_file
from src.checkpointing import save_checkpoint, load_checkpoint
from src.model_utils import create_model, setup_training, setup_metrics
from src.config import (
    TRAIN_IMG_DIR,
    TEST_IMG_DIR,
    SAMPLE_SUBMISSION_CSV_PATH,
    NUM_EPOCHS,
    BATCH_SIZE,
    DEBUG_MODE,
    GENERATE_SUBMISSION_ONLY,
    NUM_EPOCHS_DEBUG_MODE,
    CHECKPOINT_DIR,
    BEST_CHECKPOINT_FOR_SUBMISSION,
    POS_WEIGHT,
    PREDICTION_THRESHOLD_SUB)


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

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True) # Ensure checkpoint dir exists

    run_id = time.strftime('%Y%m%d_%H%M%S') # Unique ID for this training run

    train_df, val_df, pos_weight_tensor = prepare_data()

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
    if POS_WEIGHT:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))

    current_epochs = NUM_EPOCHS_DEBUG_MODE if DEBUG_MODE else NUM_EPOCHS

    start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler)
    if start_epoch > 0 : # Only print resume message if actually resuming
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("Starting training from epoch 1")

    writer = SummaryWriter(f"runs/run_{run_id}") # Use run_id for TensorBoard log dir
    
    start_time = time.time()
    for epoch in range(start_epoch, current_epochs):
        avg_train_loss, computed_metrics_train = train_epoch(
            model, train_loader, optimizer, loss_fn, metrics, device, writer, epoch
        )
        avg_val_loss, computed_metrics_val = validate_epoch(
            model, val_loader, loss_fn, metrics, device, epoch, writer
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if writer:
            writer.add_scalar("learning_rate", current_lr, epoch)

        print("\nMetrics:")
        print(
            f"Train - Loss: {avg_train_loss:.4f}, "
            f"F1: {computed_metrics_train['f1']:.4f}, "
            f"Hamming: {computed_metrics_train['hamming']:.4f}, "
            f"Exact Match: {computed_metrics_train['exact_match']:.4f}"
        )
        print(
            f"Val   - Loss: {avg_val_loss:.4f}, "
            f"F1: {computed_metrics_val['f1']:.4f}, "
            f"Hamming: {computed_metrics_val['hamming']:.4f}, "
            f"Exact Match: {computed_metrics_val['exact_match']:.4f}"
        )

        # Convert tensor metrics to Python numbers for JSON serialization
        serializable_metrics_train = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in computed_metrics_train.items()}
        serializable_metrics_val = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in computed_metrics_val.items()}

        all_epoch_metrics = {"train": serializable_metrics_train, "val": serializable_metrics_val}

        
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, all_epoch_metrics, avg_val_loss, is_best, run_id=run_id
        )
    if writer:
        writer.close()
    print(f"\nTotal training time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    if GENERATE_SUBMISSION_ONLY:
        if not BEST_CHECKPOINT_FOR_SUBMISSION.exists():
            print(f"ERROR: Checkpoint for submission not found at {BEST_CHECKPOINT_FOR_SUBMISSION}")
            print(f"Please ensure a model is trained and '{BEST_CHECKPOINT_FOR_SUBMISSION.name}' exists in '{CHECKPOINT_DIR}'.")
        else:
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            submission_name = f"submission_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            generate_submission_file(
                model_checkpoint_path=BEST_CHECKPOINT_FOR_SUBMISSION,
                device=selected_device,
                create_model_fn=create_model,
                batch_size=BATCH_SIZE,
                debug_mode=DEBUG_MODE,
                test_img_dir=TEST_IMG_DIR,
                sample_submission_csv_path=SAMPLE_SUBMISSION_CSV_PATH,
                submission_output_path=Path('submissions') / submission_name,
                prediction_threshold=PREDICTION_THRESHOLD_SUB,
            )
    else:
        main()
