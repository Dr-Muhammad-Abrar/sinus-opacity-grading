"""
Step 7 - Export ONE image per unique X-ray for blind radiologist grading.

Why: the brightness-based grades are unstable (28 of 71 twin pairs got two
different grades). A radiologist grade is a stable, independent label.
There are only 106 unique X-rays (71 train twin pairs + 24 valid + 11 test),
so one grade per X-ray is a manageable job.

What it makes in outputs/radiologist_grading/ :
  images/R001.png           the sinus crop (padded box, original resolution)
  images/R001_context.png   the whole radiograph with the box drawn (for context)
  grading_sheet.csv         BLANK sheet for the radiologist to fill in
  README_for_radiologist.txt  the grading scale
  grading_key_DO_NOT_SHARE.csv  maps R001 -> split / image ids / automated grade
GIVE THE RADIOLOGIST: the images folder + grading_sheet.csv + README only.
KEEP PRIVATE: grading_key_DO_NOT_SHARE.csv (it contains the automated grades).
"""
import sys, os, ast, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from PIL import Image, ImageDraw

OUT_DIR = os.path.join(config.OUTPUT_DIR, "radiologist_grading")
IMG_DIR = os.path.join(OUT_DIR, "images")
SEED = 42
CONTEXT_MAX_SIDE = 1600

os.makedirs(IMG_DIR, exist_ok=True)
df = pd.read_csv(config.LABELS_CSV)

# one copy per X-ray: in train, twins are (odd id, next even id) -> keep the odd one
items = []   # (exported_row, twin_row_or_None)
for split in ["train", "valid", "test"]:
    sub = df[df["split"] == split].copy()
    if split == "train":
        sub["pair"] = (sub["image_id"] + 1) // 2
        for _, g in sub.groupby("pair"):
            g = g.sort_values("image_id")
            items.append((g.iloc[0], g.iloc[1] if len(g) == 2 else None))
    else:
        for _, r in sub.iterrows():
            items.append((r, None))

random.Random(SEED).shuffle(items)          # random order, so order hints nothing

key_rows, sheet_rows = [], []
for n, (r, twin) in enumerate(items, start=1):
    sid = f"R{n:03d}"
    split = r["split"]
    x, y, w, h = ast.literal_eval(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"]
    image = Image.open(os.path.join(config.IMAGES[split], r["file_name"])).convert("L")

    # sinus crop, same padding rule as the training pipeline
    pad_x, pad_y = w * config.PAD_RATIO, h * config.PAD_RATIO
    left, top = max(0, int(x - pad_x)), max(0, int(y - pad_y))
    right, bottom = min(image.width, int(x + w + pad_x)), min(image.height, int(y + h + pad_y))
    image.crop((left, top, right, bottom)).save(os.path.join(IMG_DIR, f"{sid}.png"))

    # context: whole radiograph with the box outlined
    ctx = image.copy().convert("RGB")
    ImageDraw.Draw(ctx).rectangle([x, y, x + w, y + h], outline=(255, 255, 0),
                                  width=max(3, ctx.width // 300))
    scale = CONTEXT_MAX_SIDE / max(ctx.size)
    if scale < 1:
        ctx = ctx.resize((int(ctx.width * scale), int(ctx.height * scale)))
    ctx.save(os.path.join(IMG_DIR, f"{sid}_context.png"))

    sheet_rows.append({"sample_id": sid, "radiologist_grade": "", "notes": ""})
    key_rows.append({
        "sample_id": sid, "split": split, "image_id": int(r["image_id"]),
        "twin_image_id": int(twin["image_id"]) if twin is not None else "",
        "automated_grade": int(r["grade"]),
        "twin_automated_grade": int(twin["grade"]) if twin is not None else "",
        "mean_intensity": round(float(r["mean_intensity"]), 3), "file_name": r["file_name"],
    })

pd.DataFrame(sheet_rows).to_csv(os.path.join(OUT_DIR, "grading_sheet.csv"), index=False)
pd.DataFrame(key_rows).to_csv(os.path.join(OUT_DIR, "grading_key_DO_NOT_SHARE.csv"), index=False)

with open(os.path.join(OUT_DIR, "README_for_radiologist.txt"), "w", encoding="utf-8") as f:
    f.write("""Maxillary sinus grading - instructions

For each sample R001, R002, ... grade the maxillary sinus INSIDE the yellow box
(shown in R###_context.png). R###.png is the same region cropped with a small margin.
Fill in the 'radiologist_grade' column of grading_sheet.csv:

  0 = Normal            clear, air-filled cavity; cortical outline intact; no thickening
  1 = Mild              visible mucosal thickening or minor opacification; outline intact
  2 = Moderate/severe   marked opacification, polyp/pseudocyst appearance, or cortical disruption
  9 = Cannot grade      image quality, artifact, or the box does not show the sinus

Please grade each sample independently. Optional: write a short comment in 'notes'.
""")

print(f"Exported {len(items)} unique X-rays to: {OUT_DIR}")
print(f"  from train: {sum(1 for r,_ in items if r['split']=='train')}, "
      f"valid: {sum(1 for r,_ in items if r['split']=='valid')}, "
      f"test: {sum(1 for r,_ in items if r['split']=='test')}")
print("Give the radiologist: the 'images' folder, grading_sheet.csv, README_for_radiologist.txt")
print("Keep private: grading_key_DO_NOT_SHARE.csv")
