import torch
import os
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.config import INPUT_SIZE

class SubmissionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir
        self.model_input_channels =  ["red", "green", "blue", "yellow"]
        
        self.base_transform = T.Compose(
            [T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()]
        )
        # Normalization similar to ProteinDataset
        self.normalize = T.Normalize(mean=[0.5] * 4, std=[0.5] * 4)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]

        img_channels_data = []
        for color in self.model_input_channels:
            try:
                img_path = self.image_dir / f"{img_id}_{color}.png"
                channel_pil_img = Image.open(img_path).convert("L")
                img_channels_data.append(channel_pil_img)
            except FileNotFoundError:
                print(f"Warning: File not found {img_path}. Using a black PIL image (512x512).") # type: ignore
                img_channels_data.append(Image.new("L", (512, 512), color=0))

        # Apply base_transform (Resize, ToTensor) to each channel and then concatenate
        image_tensors = [self.base_transform(pil_img) for pil_img in img_channels_data]
        image_tensor = torch.cat(image_tensors, dim=0) # Creates a 4xHxW tensor

        normalized_image_tensor = self.normalize(image_tensor)

        return normalized_image_tensor, img_id


def generate_submission_file(
    model_checkpoint_path: Path,
    device: str,
    create_model_fn: callable, # type: ignore
    batch_size: int,
    debug_mode: bool,
    test_img_dir: Path,
    sample_submission_csv_path: Path,
    submission_output_path: Path,
    prediction_threshold: float = 0.5,
):
    print(f"Generating submission file using model: {model_checkpoint_path}")

    if not sample_submission_csv_path.exists():
        print(f"ERROR: Sample submission file not found at {sample_submission_csv_path}")
        return
    test_df = pd.read_csv(sample_submission_csv_path)

    if debug_mode:
        test_df = test_df.sample(n=min(len(test_df), 100), random_state=42)
        print(f"DEBUG_MODE: Using a small sample of test_df: {len(test_df)} images.")


    submission_dataset = SubmissionDataset(
        df=test_df,
        image_dir=test_img_dir
    )
    submission_loader = DataLoader(
        submission_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=8,
    )

    model = create_model_fn()
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictions_list = []
    image_ids_list = []

    with torch.no_grad():
        for images, ids_batch in tqdm(submission_loader, desc="Generating Predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds_batch_indices = (probs > prediction_threshold).int()

            for i in range(preds_batch_indices.shape[0]):
                pred_indices_str = [str(j) for j, label in enumerate(preds_batch_indices[i]) if label == 1]
                predictions_list.append(" ".join(pred_indices_str))
                image_ids_list.append(ids_batch[i])

    submission_df = pd.DataFrame({"Id": image_ids_list, "Predicted": predictions_list})
    submission_df.to_csv(submission_output_path, index=False)
    print(f"Submission file saved to {submission_output_path}")