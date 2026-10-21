"""
Stage 5 — Grad-CAM++ explainability.

For every test-set image, generates a Grad-CAM++ heatmap from the
trained ConvNeXt-Tiny model and saves an overlay image showing where
the model is "looking" when it makes its prediction.

Beyond the pictures, this also computes a QUANTIFIED localization
metric: since each saved ROI crop includes padding around the actual
annotated sinus bounding box, we know exactly where that original box
sits within the 224x224 crop. We measure the IoU between Grad-CAM++'s
high-activation region and that known box — i.e., does the model
actually focus on the annotated anatomical region, or is it looking at
irrelevant padding/surrounding structures? This number goes in your
paper, not just the pictures.

Saves:
    outputs/figures/gradcam_<image_id>_<true>_<pred>.png   (overlays)
    outputs/labels/gradcam_iou_scores.csv                   (per-image IoU)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import models, transforms
from coco_lite import COCO
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IOU_THRESHOLD = 0.5  # fraction of max CAM intensity used to binarize the heatmap


def build_model_and_load(ckpt_path):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def compute_core_box_in_crop(meta_row, img_size):
    """Computes where the ORIGINAL annotated sinus box sits within the
    final resized crop, correctly accounting for any rows stripped from
    the top during ruler-artifact removal (03_roi_extraction.py). Using
    the ideal padding geometry alone (ignoring stripping) silently
    misaligns this reference box for any image where stripping
    occurred, which would make the IoU metric compare against the wrong
    region -- this reads the actual recorded stripping amount instead."""
    w, h = meta_row["bbox_w"], meta_row["bbox_h"]
    pad_x, pad_y = meta_row["pad_x"], meta_row["pad_y"]
    pre_strip_w, pre_strip_h = meta_row["pre_strip_w"], meta_row["pre_strip_h"]
    band_end = meta_row["band_end"]

    post_strip_h = pre_strip_h - band_end
    post_strip_w = pre_strip_w  # stripping only removes rows, not columns

    # Box position relative to the padded crop's top-left, before stripping
    rel_x = pad_x
    rel_y = pad_y - band_end  # shifts up as rows are removed from the top

    # If the band ate into the annotated box itself, clip and shrink the
    # visible box height accordingly rather than reporting a negative position
    visible_h = h
    if rel_y < 0:
        visible_h = h + rel_y  # reduce visible height by the overlap amount
        rel_y = 0
        visible_h = max(visible_h, 0)

    scale_x = img_size / post_strip_w
    scale_y = img_size / post_strip_h if post_strip_h > 0 else 0

    core_left = rel_x * scale_x
    core_top = rel_y * scale_y
    core_right = core_left + w * scale_x
    core_bottom = core_top + visible_h * scale_y
    return int(core_left), int(core_top), int(core_right), int(core_bottom)


def iou_with_box(cam, box, threshold=IOU_THRESHOLD):
    """IoU between the thresholded high-activation CAM region and a
    rectangular box, both in the same pixel coordinate space."""
    left, top, right, bottom = box
    h, w = cam.shape
    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[max(0, top):min(h, bottom), max(0, left):min(w, right)] = True

    cam_mask = cam >= (threshold * cam.max())

    intersection = np.logical_and(cam_mask, box_mask).sum()
    union = np.logical_or(cam_mask, box_mask).sum()
    return intersection / union if union > 0 else 0.0


def main():
    ckpt_path = os.path.join(config.MODELS_DIR, "convnext_tiny_final.pt")
    model = build_model_and_load(ckpt_path)

    target_layers = [model.features[-1]]
    cam_algo = GradCAMPlusPlus(model=model, target_layers=target_layers)

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    meta_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "roi_crop_metadata.csv")
    metadata = pd.read_csv(meta_csv)
    metadata_test = metadata[metadata["split"] == "test"].set_index("image_id")

    coco = COCO(config.ANNOTATIONS["test"])
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    sinus_id = next(k for k, v in cats.items() if v.strip().lower() == config.SINUS_CLASS_NAME)

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    results = []

    for img_id in sorted(coco.getImgIds(catIds=[sinus_id])):
        crop_path_candidates = [
            os.path.join(config.ROI_DIR, "test", g, f"{img_id}.png") for g in config.GRADE_LABELS
        ]
        crop_path = next((p for p in crop_path_candidates if os.path.exists(p)), None)
        if crop_path is None:
            continue
        true_grade = int(os.path.basename(os.path.dirname(crop_path)))

        if img_id not in metadata_test.index:
            print(f"WARNING: no crop metadata for image {img_id}, skipping (re-run 03_roi_extraction.py)")
            continue
        meta_row = metadata_test.loc[img_id]

        pil_img = Image.open(crop_path).convert("RGB")
        rgb_img = np.array(pil_img.resize((config.IMG_SIZE, config.IMG_SIZE))).astype(np.float32) / 255.0
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_grade = model(input_tensor).argmax(1).item()

        cam_map = cam_algo(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(pred_grade)])[0]

        core_box = compute_core_box_in_crop(meta_row, config.IMG_SIZE)
        iou = iou_with_box(cam_map, core_box)

        overlay = show_cam_on_image(rgb_img, cam_map, use_rgb=True)
        cv2.rectangle(overlay, (core_box[0], core_box[1]), (core_box[2], core_box[3]), (255, 255, 0), 2)
        out_path = os.path.join(config.FIGURES_DIR, f"gradcam_{img_id}_true{true_grade}_pred{pred_grade}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        results.append({
            "image_id": img_id, "true_grade": true_grade, "pred_grade": pred_grade,
            "correct": true_grade == pred_grade, "iou_with_sinus_box": iou,
        })
        print(f"Image {img_id}: true={true_grade} pred={pred_grade} IoU={iou:.3f}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "gradcam_iou_scores.csv")
    df.to_csv(out_csv, index=False)

    print(f"\nSaved {len(df)} overlays to: {config.FIGURES_DIR}")
    print(f"Saved IoU scores to: {out_csv}")
    print(f"\nMean IoU (all images): {df['iou_with_sinus_box'].mean():.3f}")
    print(f"Mean IoU (correct predictions): {df[df['correct']]['iou_with_sinus_box'].mean():.3f}")
    if (~df["correct"]).any():
        print(f"Mean IoU (incorrect predictions): {df[~df['correct']]['iou_with_sinus_box'].mean():.3f}")


if __name__ == "__main__":
    main()
