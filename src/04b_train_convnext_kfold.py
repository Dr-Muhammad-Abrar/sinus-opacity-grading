"""
Stage 4b - ConvNeXt-Tiny with twin-grouped 5-fold cross-validation.

Pools outputs/rois/train + outputs/rois/valid (166 images after cleaning) and runs
stratified 5-fold cross-validation on GROUPS. In the train split of this dataset
export, copies of one radiograph have consecutive image ids (odd id, next even id).
They form a "twin pair" and always stay in the same fold (see group_id). Check this
assumption with 11_check_twins.py before using other data.

Within each fold, 20% of the training groups are held out as an inner validation
set that is used ONLY for early stopping. The scored fold is never used to choose
an epoch. The original test split (11 images) is not touched during cross-validation.
The final model is trained on an inner split of the whole pool and the test set is
scored once, at the end.

Saves:
    outputs/models/convnext_tiny_final.pt
    outputs/labels/cnn_predictions.csv       final test predictions
    outputs/labels/kfold_results.csv         per-fold metrics
"""

import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from simple_metrics import accuracy_score, f1_score, confusion_matrix
def spearman_corr(x, y):
    """Pure-NumPy Spearman correlation with a normal-approximation p-value
    -- avoids scipy.stats, whose import chain (via scipy.spatial) has been
    repeatedly blocked by this machine's Windows Application Control policy."""
    x, y = np.asarray(x), np.asarray(y)
    def rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    rx, ry = rank(x), rank(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    if n > 3 and abs(rho) < 1:
        t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    else:
        p_value = float("nan")
    return rho, p_value


def manual_stratified_kfold(labels, n_splits, seed=42):
    """Pure-NumPy stratified k-fold split (avoids sklearn.model_selection,
    whose compiled _random module can be blocked by Windows Application
    Control policies on some locked-down machines). Returns a list of
    (train_idx, val_idx) index arrays, one per fold."""
    labels = np.array(labels)
    rng = np.random.RandomState(seed)

    # For each class, shuffle its indices and split into n_splits chunks
    class_folds = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        class_folds[c] = np.array_split(idx, n_splits)

    folds = []
    for k in range(n_splits):
        val_idx = np.concatenate([class_folds[c][k] for c in class_folds])
        train_idx = np.concatenate([
            np.concatenate([class_folds[c][j] for j in range(n_splits) if j != k])
            for c in class_folds
        ])
        folds.append((train_idx, val_idx))
    return folds


def group_id(path):
    """Twin-safe key. In the train split, copies of the same X-ray have
    consecutive ids (odd id, next even id), e.g. 8413/8414, so both get the
    same key. Valid images are single."""
    p = path.replace("\\", "/").split("/")
    split, img_id = p[-3], int(p[-1].split(".")[0])
    return (split, (img_id + 1) // 2) if split == "train" else (split, img_id)


def grouped_stratified_kfold(samples, n_splits, seed=42):
    """Stratified k-fold that keeps each twin pair in the same fold."""
    rng = np.random.RandomState(seed)
    groups = {}
    for i, (path, label) in enumerate(samples):
        groups.setdefault(group_id(path), []).append(i)
    keys = list(groups)
    glabels = np.array([np.bincount([samples[i][1] for i in groups[k]]).argmax()
                        for k in keys])
    mixed = sum(len({samples[i][1] for i in groups[k]}) > 1 for k in keys)
    print(f"groups: {len(keys)} | groups with mixed labels: {mixed}")

    class_folds = {}
    for c in np.unique(glabels):
        idx = np.where(glabels == c)[0]
        rng.shuffle(idx)
        class_folds[c] = np.array_split(idx, n_splits)

    folds = []
    for k in range(n_splits):
        val_g = np.concatenate([class_folds[c][k] for c in class_folds])
        tr_g = np.concatenate([
            np.concatenate([class_folds[c][j] for j in range(n_splits) if j != k])
            for c in class_folds])
        folds.append((np.array([i for g in tr_g for i in groups[keys[g]]]),
                      np.array([i for g in val_g for i in groups[keys[g]]])))
    return folds


def split_inner(samples, frac=0.2, seed=42):
    """Splits (path, label) samples into inner-train / inner-val, keeping twin
    pairs together and stratifying by class. The inner-val part is used ONLY
    to decide when to stop training / which epoch to keep. The data that is
    later used for scoring (the CV fold, or the test set) never takes part."""
    rng = np.random.RandomState(seed)
    groups = {}
    for s in samples:
        groups.setdefault(group_id(s[0]), []).append(s)
    by_class = {}
    for k, items in groups.items():
        lab = int(np.bincount([l for _, l in items]).argmax())
        by_class.setdefault(lab, []).append(k)
    val_keys = set()
    for lab, keys in by_class.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = max(1, int(round(len(keys) * frac))) if len(keys) > 1 else 0
        val_keys.update(keys[:n])
    inner_tr = [s for k, items in groups.items() if k not in val_keys for s in items]
    inner_val = [s for k in val_keys for s in groups[k]]
    return inner_tr, inner_val

# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 25
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42

def set_all_seeds(seed=SEED):
    """Fixes every source of randomness (Python, NumPy, PyTorch, CUDA)
    so re-running this script produces the same trained model and the
    same downstream Grad-CAM/IoU results, rather than a new random draw
    each time -- essential once you're locking in numbers to report."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ------------------------------------------------------------------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class ListDataset(Dataset):
    """Wraps a plain list of (path, label) so we can apply different
    transforms to arbitrary train/val subsets drawn from a combined pool."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


def train_one_run(train_samples, val_samples, train_tf, eval_tf, max_epochs=EPOCHS):
    """Trains a fresh model with early stopping. Used for both CV folds
    and the final full-pool training run."""
    class_counts = np.bincount([label for _, label in train_samples])
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(DEVICE)

    train_loader = DataLoader(ListDataset(train_samples, train_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ListDataset(val_samples, eval_tf), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_f1, best_state, no_improve = -1, None, 0
    for epoch in range(1, max_epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion)
        scheduler.step()
        if val_f1 > best_f1:
            best_f1, best_state, no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_f1


def collect_samples(split_dir):
    ds = datasets.ImageFolder(split_dir)
    return ds.samples, ds.classes


def main():
    set_all_seeds()
    train_tf, eval_tf = build_transforms()

    train_samples, classes = collect_samples(os.path.join(config.ROI_DIR, "train"))
    valid_samples, _ = collect_samples(os.path.join(config.ROI_DIR, "valid"))
    test_samples, _ = collect_samples(os.path.join(config.ROI_DIR, "test"))
    pool_samples = train_samples + valid_samples
    print(f"Cross-validation pool: {len(pool_samples)} images | held-out final test: {len(test_samples)} images")
    print("Classes:", classes)

    labels = [label for _, label in pool_samples]
    folds = grouped_stratified_kfold(pool_samples, N_FOLDS, seed=SEED)
    for tr, va in folds:   # safety check: no twin pair may straddle a fold
        assert not ({group_id(pool_samples[i][0]) for i in tr} &
                    {group_id(pool_samples[i][0]) for i in va}), "twin leak!"

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_train = [pool_samples[i] for i in train_idx]
        fold_val = [pool_samples[i] for i in val_idx]

        inner_tr, inner_val = split_inner(fold_train, seed=SEED + fold)
        print(f"Fold {fold}: inner-train {len(inner_tr)} / inner-val {len(inner_val)} "
              f"(only for early stopping) | scored fold: {len(fold_val)}")
        model, best_val_f1 = train_one_run(inner_tr, inner_val, train_tf, eval_tf)

        val_loader = DataLoader(ListDataset(fold_val, eval_tf), batch_size=BATCH_SIZE, shuffle=False)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                out = model(imgs.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(lbls.numpy())

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")
        fold_results.append({"fold": fold, "n_val": len(fold_val), "accuracy": acc, "macro_f1": f1})
        print(f"Fold {fold}: n_val={len(fold_val)} | accuracy={acc:.3f} | macro_f1={f1:.3f}")

    results_df = pd.DataFrame(fold_results)
    results_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "kfold_results.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n===== Cross-validation summary (5 folds, every image tested once) =====")
    print(f"Accuracy: {results_df['accuracy'].mean():.3f} +/- {results_df['accuracy'].std():.3f}")
    print(f"Macro F1: {results_df['macro_f1'].mean():.3f} +/- {results_df['macro_f1'].std():.3f}")
    print(f"Saved per-fold results to: {results_csv}")

    # ---------------- Final model: train on the pooled images (inner split for early stopping) ----------------
    print(f"\nTraining final model on the full {len(pool_samples)}-image pool...")
    inner_tr, inner_val = split_inner(pool_samples, seed=SEED)
    print(f"Final model: inner-train {len(inner_tr)} / inner-val {len(inner_val)} (only for early stopping)")
    final_model, final_val_f1 = train_one_run(inner_tr, inner_val, train_tf, eval_tf)
    # The test set is NOT used for early stopping or model choice; it is scored once, below.

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.MODELS_DIR, "convnext_tiny_final.pt")
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"Saved final model to: {ckpt_path}")

    # ---------------- Final, single evaluation on the untouched test set ----------------
    test_loader = DataLoader(ListDataset(test_samples, eval_tf), batch_size=BATCH_SIZE, shuffle=False)
    final_model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            out = final_model(imgs.to(DEVICE))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    rho, p_value = spearman_corr(all_labels, all_preds)

    print(f"\n===== FINAL held-out test results ({len(test_samples)} images, evaluated once) =====")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(f"Spearman correlation (predicted vs. ground-truth grade): rho={rho:.3f}, p={p_value:.4f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    pred_rows = []
    for (path, true_label), pred_label, probs in zip(test_samples, all_preds, all_probs):
        image_id = os.path.splitext(os.path.basename(path))[0]
        pred_rows.append({
            "split": "test", "image_id": image_id,
            "true_grade": true_label, "pred_grade": pred_label,
            "prob_0": probs[0], "prob_1": probs[1], "prob_2": probs[2],
        })
    pred_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "cnn_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)
    print(f"\nSaved final test predictions to: {pred_csv}")

    print("\n=== For your paper, report BOTH numbers ===")
    print(f"Cross-validated estimate (robust): accuracy {results_df['accuracy'].mean():.3f} "
          f"+/- {results_df['accuracy'].std():.3f}, macro F1 {results_df['macro_f1'].mean():.3f} "
          f"+/- {results_df['macro_f1'].std():.3f}  (n={len(pool_samples)}, 5-fold)")
    print(f"Final held-out test (single measurement): accuracy {acc:.3f}, macro F1 {f1:.3f}  (n={len(test_samples)})")


if __name__ == "__main__":
    main()
