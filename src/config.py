from pathlib import Path

# Base Data Paths
CUSTOM_DATA_PATH = Path('data')
TRAIN_CSV_PATH = CUSTOM_DATA_PATH / "train.csv"
TRAIN_IMG_DIR = CUSTOM_DATA_PATH / "train"
TEST_IMG_DIR = CUSTOM_DATA_PATH / "test"
SAMPLE_SUBMISSION_CSV_PATH = CUSTOM_DATA_PATH / "sample_submission.csv"
SUBMISSION_OUTPUT_PATH = Path("submission.csv")

# Model & Training Parameters
NUM_CLASSES = 28
NUM_EPOCHS = 100
BATCH_SIZE = 64
FREEZE_BACKBONE = True
MAX_GRAD_NORM = 0.4
POS_WEIGHT = False
"""
Vit Models:
vit_small_patch14_dinov2.lvd142m
tiny_vit_21m_224
vit_base_patch16_384
"""
PRETRAIND_MODEL =  "tiny_vit_21m_224"
INPUT_SIZE = 224 #! Input image size for model!

# Debugging & Checkpointing
DEBUG_MODE = False
NUM_EPOCHS_DEBUG_MODE = 5 
RESUME_TRAINING = False
DISABLE_CHECKPOINTS_IN_DEBUG = True
# Submission Configuration
GENERATE_SUBMISSION_ONLY = False
PREDICTION_THRESHOLD_SUB = 0.5

CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT_FOR_SUBMISSION = CHECKPOINT_DIR / "20250614_125118_latest_checkpoint.pt"
