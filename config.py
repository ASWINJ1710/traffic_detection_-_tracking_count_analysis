from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "data" / "models" / "best.pt"
VIDEO_DIR = BASE_DIR / "data" / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"

# Vehicle classes
CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorbike",
}

# YOLO detection settings
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = 0

# Tracking settings
TRACKER = "bytetrack.yaml"

# Line crossing settings
COUNTING_LINE_POSITION = 0.60

# Video settings
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Create required directories
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
