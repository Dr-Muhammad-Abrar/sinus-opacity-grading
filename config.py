"""
Central configuration. Every script imports its paths and settings from here.

Set the dataset location once, either by editing DATASET_ROOT below or (recommended)
by setting the environment variable SINUS_DATASET_ROOT to the folder that contains
annotations/, train/, valid/ and test/ (the COCO export of the Kaggle dataset).

    Windows (PowerShell):  $env:SINUS_DATASET_ROOT = "D:\\data\\COCO"
    Linux / macOS:         export SINUS_DATASET_ROOT=/data/COCO
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 1. RAW DATASET (COCO format only; the YOLO export is not used)
# ------------------------------------------------------------------
DATASET_ROOT = os.environ.get("SINUS_DATASET_ROOT", os.path.join(PROJECT_ROOT, "data", "COCO"))

ANNOTATIONS = {
    "train": os.path.join(DATASET_ROOT, "annotations", "train_coco.json"),
    "valid": os.path.join(DATASET_ROOT, "annotations", "valid_coco.json"),
    "test": os.path.join(DATASET_ROOT, "annotations", "test_coco.json"),
}

IMAGES = {
    "train": os.path.join(DATASET_ROOT, "train"),
    "valid": os.path.join(DATASET_ROOT, "valid"),
    "test": os.path.join(DATASET_ROOT, "test"),
}

SINUS_CLASS_NAME = "maxillary sinus"

# ------------------------------------------------------------------
# 2. OUTPUTS (kept separate from the raw dataset; git-ignored)
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# The file name keeps the word "severity" for historical reasons; the paper calls
# these grades "opacity grades".
LABELS_CSV = os.path.join(OUTPUT_DIR, "labels", "sinus_severity_labels.csv")
ROI_DIR = os.path.join(OUTPUT_DIR, "rois")          # cropped sinus images, by split/grade
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")     # trained checkpoints
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")   # Grad-CAM++ overlays, ODE plots

# ------------------------------------------------------------------
# 3. GRADES: tertiles of an automated brightness score (not clinically validated)
# ------------------------------------------------------------------
GRADE_LABELS = {
    "0": "Low",
    "1": "Intermediate",
    "2": "High",
}

# ------------------------------------------------------------------
# 4. ROI EXTRACTION
# ------------------------------------------------------------------
PAD_RATIO = 0.15   # padding around each sinus box, as a fraction of box size
IMG_SIZE = 224     # resize target for the network input
