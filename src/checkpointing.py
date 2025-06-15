import torch
import json
from datetime import datetime
from src.config import (
    CHECKPOINT_DIR,
    DEBUG_MODE,
    DISABLE_CHECKPOINTS_IN_DEBUG,
    RESUME_TRAINING,
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    FREEZE_BACKBONE,
    MAX_GRAD_NORM,
    POS_WEIGHT,
    NUM_EPOCHS_DEBUG_MODE,
    GENERATE_SUBMISSION_ONLY,
    PRETRAIND_MODEL,
    INPUT_SIZE
)

def save_checkpoint(model, optimizer, scheduler, epoch, metrics_dict, loss, is_best=False, run_id=None):
    """Saves model checkpoint and metrics history."""
    if DISABLE_CHECKPOINTS_IN_DEBUG and DEBUG_MODE:
        print("INFO: DEBUG_MODE & DISABLE_CHECKPOINTS_IN_DEBUG are True. Checkpoint saving skipped.")
        return

    if run_id is None:
        # Fallback if run_id is not provided, though it should be from main.py
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_defaultRun')
        print(f"WARNING: run_id not provided to save_checkpoint, using default: {run_id}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "metrics": metrics_dict,
    }

    # --- Checkpoint .pt files ---
    # These files are timestamped for each save, allowing load_checkpoint to find the most recent.
    now = datetime.now()
    current_save_timestamp = now.strftime("%Y%m%d_%H%M%S_%f") # Precise timestamp (YearMonthDay_HourMinSec_Microsecond)

    # Save the latest checkpoint with a unique timestamped name
    # 1. Save/Overwrite the latest checkpoint for the current run
    # This file is always updated with the state of the last saved epoch for this run.
    run_latest_checkpoint_path = CHECKPOINT_DIR / f"{run_id}_latest_checkpoint.pt"
    torch.save(checkpoint, run_latest_checkpoint_path)

    if is_best:
        # Also save this best state with a unique timestamped name (for history of bests)
        run_best_checkpoint_path = CHECKPOINT_DIR / f"{run_id}_best_checkpoint.pt"
        torch.save(checkpoint, run_best_checkpoint_path)

        # Overwrite the fixed 'best_checkpoint.pt' used for submission (from config.py)
        fixed_best_checkpoint_path = CHECKPOINT_DIR / "best_checkpoint.pt"
        torch.save(checkpoint, fixed_best_checkpoint_path)
    
    # --- Metrics history file (one per run_id, append mode) ---
    metrics_file_path = CHECKPOINT_DIR / f"{run_id}_metrics_history.json"
    history = []
    if metrics_file_path.exists():
        with metrics_file_path.open("r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = [] # Start fresh if file is corrupted

    history.append({"epoch": epoch, "loss": loss, "metrics": metrics_dict})
    with metrics_file_path.open("w") as f: # Overwrite with updated history
        json.dump(history, f, indent=2)

    # --- Settings file (one per run_id, write once) ---
    settings_file_path = CHECKPOINT_DIR / f"{run_id}_settings.txt"
    if not settings_file_path.exists(): # Create only if it doesn't exist for this run_id
        with open(settings_file_path, "w") as f:
            f.write(f"Run ID: {run_id}\n")
            f.write(f"PRETRAINED_MODEL = {PRETRAIND_MODEL}\nINPUT_SIZE = {INPUT_SIZE}\n\n" # Add model info
                    f"NUM_CLASSES = {NUM_CLASSES}\nNUM_EPOCHS = {NUM_EPOCHS}\nBATCH_SIZE = {BATCH_SIZE}\nFREEZE_BACKBONE = {FREEZE_BACKBONE}\n"
                    f"MAX_GRAD_NORM = {MAX_GRAD_NORM}\nPOS_WEIGHT = {POS_WEIGHT}\nDEBUG_MODE = {DEBUG_MODE}\nNUM_EPOCHS_DEBUG_MODE = {NUM_EPOCHS_DEBUG_MODE}\n"
                    f"RESUME_TRAINING = {RESUME_TRAINING}\nDISABLE_CHECKPOINTS_IN_DEBUG = {DISABLE_CHECKPOINTS_IN_DEBUG}\nGENERATE_SUBMISSION_ONLY = {GENERATE_SUBMISSION_ONLY}\n"
                    f"PRETRAIND_MODEL = {PRETRAIND_MODEL}\nINPUT_SIZE = {INPUT_SIZE}\n")


def load_checkpoint(model, optimizer, scheduler):
    """Loads model checkpoint."""
    if DISABLE_CHECKPOINTS_IN_DEBUG and DEBUG_MODE:
        print("INFO: DEBUG_MODE & DISABLE_CHECKPOINTS_IN_DEBUG are True. Checkpoint loading skipped.")
        return 0, float("inf")
    if not RESUME_TRAINING:
        print("INFO: RESUME_TRAINING is False. Starting new training run.")
        return 0, float("inf")

    candidate_files = list(CHECKPOINT_DIR.glob('*_latest_checkpoint.pt'))

    if not candidate_files:
        print(f"INFO: No '..._latest_checkpoint.pt' files found in {CHECKPOINT_DIR}. Starting new training run.")
        return 0, float("inf")

    # Sort files by modification time in descending order (newest first)
    candidate_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    checkpoint_path = candidate_files[0] # The most recent one

    print(f"INFO: Attempting to load checkpoint: {checkpoint_path.name}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint.get("loss", float("inf"))
        print(f"INFO: Resumed training from checkpoint {checkpoint_path.name}. Epoch: {epoch}, Loss: {loss:.4f}")
        return epoch, loss
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint {checkpoint_path}. Error: {e}. Starting new training run.")
        return 0, float("inf")