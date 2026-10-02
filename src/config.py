"""Central configuration for VisionTalk.

Paths are relative to the project root. Place Flickr8k under data/raw/ manually;
nothing is downloaded automatically. Increase EPOCHS to 20–50 only after the
pipeline works on your machine.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Directories ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_FEATURES_DIR = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Preferred Flickr8k layout (original academic distribution)
DATASET_PATH = DATA_RAW_DIR / "Flickr8k_Dataset"
FLICKR8K_TEXT_DIR = DATA_RAW_DIR / "Flickr8k_text"
CAPTION_FILE = FLICKR8K_TEXT_DIR / "Flickr8k.token.txt"
TRAIN_IMAGES_FILE = FLICKR8K_TEXT_DIR / "Flickr_8k.trainImages.txt"
VAL_IMAGES_FILE = FLICKR8K_TEXT_DIR / "Flickr_8k.devImages.txt"
TEST_IMAGES_FILE = FLICKR8K_TEXT_DIR / "Flickr_8k.testImages.txt"

# Alternate Kaggle-style layout
ALT_FLICKR8K_DIR = DATA_RAW_DIR / "Flickr8k"
ALT_IMAGES_DIR = ALT_FLICKR8K_DIR / "Images"
ALT_CAPTION_FILE = ALT_FLICKR8K_DIR / "captions.txt"

# Defaults used by resolvers (overridden if an alternate layout is found)
FLICKR8K_DIR = DATASET_PATH
FLICKR8K_IMAGES_DIR = DATASET_PATH
FLICKR8K_CAPTIONS_FILE = CAPTION_FILE

# Cached artifacts
PROCESSED_CAPTIONS_PATH = DATA_PROCESSED_DIR / "captions.json"
SEQUENCES_PATH = DATA_PROCESSED_DIR / "sequences.json"
TOKENIZER_PATH = DATA_PROCESSED_DIR / "tokenizer.pkl"
METADATA_PATH = DATA_PROCESSED_DIR / "metadata.json"
FEATURES_CACHE_PATH = DATA_FEATURES_DIR / "image_features.pkl"
FEATURES_VECTOR_DIR = DATA_FEATURES_DIR / "vectors"
FEATURES_METADATA_PATH = DATA_FEATURES_DIR / "extractor_metadata.json"
SPLIT_PATH = DATA_PROCESSED_DIR / "splits.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CHECKPOINT_PATH = CHECKPOINTS_DIR / "latest_caption_model.keras"
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_caption_model.keras"
MODEL_PATH = FINAL_MODEL_DIR / "caption_model.keras"
FINAL_MODEL_PATH = MODEL_PATH
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
LOSS_PLOT_PATH = FIGURES_DIR / "training_loss.png"

# --- Text ---
START_TOKEN = "startseq"
END_TOKEN = "endseq"
PAD_TOKEN = "pad"
UNK_TOKEN = "unk"
VOCAB_SIZE = None  # filled after the tokenizer is built
MAX_CAPTION_LENGTH = None  # computed from training captions
MAX_VOCAB_SIZE = None  # None = keep all training words; set e.g. 10000 to cap

# --- Image / CNN (VGG16 per internship Week 1 requirement) ---
CNN_MODEL_NAME = "VGG16"
CNN_WEIGHTS = "imagenet"
CNN_POOLING = None
IMAGE_SIZE = (224, 224)
FEATURE_VECTOR_DIM = 4096  # VGG16 fc2 (penultimate) layer output (include_top=True)
FEATURE_BATCH_SIZE = 16

# --- Training ---
# 20–50 epochs required by internship final training phase.
BATCH_SIZE = 32
EPOCHS = 30
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DENSE_UNITS = 256
DROPOUT = 0.3
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MISSING_IMAGE_RAISE_COUNT = 50
MISSING_IMAGE_RAISE_FRACTION = 0.20

# --- Inference / evaluation ---
BEAM_SIZE = 3
EVAL_SAMPLE_IMAGES = 10


def ensure_project_directories() -> None:
    """Create empty project folders if they are missing."""
    for directory in (
        DATA_RAW_DIR,
        DATASET_PATH,
        FLICKR8K_TEXT_DIR,
        ALT_IMAGES_DIR,
        DATA_PROCESSED_DIR,
        DATA_FEATURES_DIR,
        FEATURES_VECTOR_DIR,
        CHECKPOINTS_DIR,
        FINAL_MODEL_DIR,
        NOTEBOOKS_DIR,
        FIGURES_DIR,
        EVALUATION_DIR,
        PREDICTIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _dir_has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    try:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def _nested_image_dir(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    if _dir_has_images(directory):
        return directory
    try:
        children = sorted(p for p in directory.iterdir() if p.is_dir())
    except OSError:
        return None
    for child in children:
        if _dir_has_images(child):
            return child
    return None


def resolve_images_dir() -> Path:
    """Locate Flickr8k images under common folder names."""
    for candidate in (
        DATASET_PATH,
        ALT_IMAGES_DIR,
        ALT_FLICKR8K_DIR,
        DATA_RAW_DIR / "Flicker8k_Dataset",
        DATA_RAW_DIR / "Flickr8k_Dataset" / "Flicker8k_Dataset",
    ):
        found = _nested_image_dir(candidate)
        if found is not None:
            return found
    return DATASET_PATH


def resolve_caption_file() -> Path:
    """Locate Flickr8k.token.txt or captions.txt under common folder names."""
    for candidate in (
        CAPTION_FILE,
        FLICKR8K_TEXT_DIR / "Flickr8k.token.txt",
        ALT_CAPTION_FILE,
        DATA_RAW_DIR / "Flickr8k" / "Flickr8k.token.txt",
        DATA_RAW_DIR / "Flickr8k_text" / "captions.txt",
    ):
        if candidate.is_file():
            return candidate
    return CAPTION_FILE


def dataset_missing_message() -> str:
    return (
        "Flickr8k dataset not found. Place the dataset in data/raw/ according to "
        "README.md and rerun preprocessing."
    )
