import torch
from tqdm import tqdm
from src.config import MAX_GRAD_NORM

def train_epoch(model, train_loader, optimizer, loss_fn, metrics, device, writer, epoch):
    """Runs one training epoch."""
    model.train()
    total_loss = 0
    metrics["train"].reset()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}")
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
        metrics["train"].update(preds, labels_int)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if writer:
            writer.add_scalar("train/batch_loss", loss.item(), epoch * len(train_loader) + batch_idx)

    avg_loss = total_loss / len(train_loader)
    computed_metrics_train = metrics["train"].compute()
    if writer:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        for k, v in computed_metrics_train.items():
            writer.add_scalar(f"train/{k}", v, epoch)
    
    return avg_loss, computed_metrics_train


def validate_epoch(model, val_loader, loss_fn, metrics, device, epoch, writer):
    """Runs one validation epoch."""
    model.eval()
    total_loss = 0
    metrics["val"].reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch+1}")
        for images, labels in pbar:
            images = images.to(device)
            labels_float = labels.to(device).float()
            labels_int = labels.to(device).int()

            outputs = model(images)
            loss = loss_fn(outputs, labels_float)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            metrics["val"].update(preds, labels_int)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_loader)
    computed_metrics_val = metrics["val"].compute()
    if writer:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)
        for k, v in computed_metrics_val.items():
            writer.add_scalar(f"val/{k}", v, epoch)
            
    return avg_loss, computed_metrics_val