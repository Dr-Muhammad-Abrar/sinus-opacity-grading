"""
Stage 2 — Manual severity grading tool.
Shows each sinus ROI (across all three splits) one at a time and lets
you assign a severity grade with a keypress. Saves to config.LABELS_CSV,
resumes automatically if you stop partway through.

Keys: 0/1/2 = grade + next   b = back   s = skip   q = save & quit
"""

import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from tkinter import Tk, Canvas, Label
from PIL import Image, ImageTk
from coco_lite import COCO

DISPLAY_SIZE = 500


def load_already_graded(csv_path):
    graded = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                graded[(row["split"], int(row["image_id"]))] = row
    return graded


def append_row(csv_path, row, write_header):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "image_id", "file_name", "bbox", "grade", "grade_label"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_task_list():
    """Collect every sinus image across all three splits into one queue."""
    tasks = []
    coco_objs = {}
    for split in ["train", "valid", "test"]:
        coco = COCO(config.ANNOTATIONS[split])
        coco_objs[split] = coco
        cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        sinus_id = next((k for k, v in cats.items() if v.strip().lower() == config.SINUS_CLASS_NAME), None)
        if sinus_id is None:
            continue
        for img_id in sorted(coco.getImgIds(catIds=[sinus_id])):
            tasks.append((split, img_id, sinus_id))
    return tasks, coco_objs


def main():
    tasks, coco_objs = build_task_list()
    already = load_already_graded(config.LABELS_CSV)
    remaining = [t for t in tasks if (t[0], t[1]) not in already]

    print(f"Total sinus images: {len(tasks)} | already graded: {len(already)} | remaining: {len(remaining)}")
    if not remaining:
        print("Nothing left to grade.")
        return

    write_header = not os.path.exists(config.LABELS_CSV)
    state = {"idx": 0, "tasks": remaining, "history": []}

    root = Tk()
    root.title("Sinus severity grading")
    canvas = Canvas(root, width=DISPLAY_SIZE, height=DISPLAY_SIZE)
    canvas.pack()
    info = Label(root, text="", font=("Segoe UI", 12), justify="left")
    info.pack(pady=6)
    Label(
        root,
        text="Press 0=Normal  1=Mild  2=Moderate/severe   |   b=back  s=skip  q=save & quit",
        font=("Segoe UI", 11),
    ).pack(pady=4)

    tk_img_ref = {"img": None}

    def crop_current():
        split, img_id, sinus_id = state["tasks"][state["idx"]]
        coco = coco_objs[split]
        img_info = coco.loadImgs([img_id])[0]
        ann = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=[sinus_id]))[0]
        x, y, w, h = ann["bbox"]
        pad_x, pad_y = w * config.PAD_RATIO, h * config.PAD_RATIO

        img_path = os.path.join(config.IMAGES[split], img_info["file_name"])
        image = Image.open(img_path).convert("L")
        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(image.width, int(x + w + pad_x))
        bottom = min(image.height, int(y + h + pad_y))
        crop = image.crop((left, top, right, bottom))
        crop.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
        return split, img_id, img_info["file_name"], ann["bbox"], crop

    def show_current():
        if state["idx"] >= len(state["tasks"]):
            info.config(text="All done! CSV saved.")
            canvas.delete("all")
            return
        split, img_id, file_name, bbox, crop = crop_current()
        tk_img = ImageTk.PhotoImage(crop)
        tk_img_ref["img"] = tk_img
        canvas.delete("all")
        canvas.create_image(DISPLAY_SIZE // 2, DISPLAY_SIZE // 2, image=tk_img)
        remaining_count = len(state["tasks"]) - state["idx"]
        info.config(text=f"[{split}] Image ID {img_id}  |  {file_name}  |  {remaining_count} remaining")
        state["current"] = (split, img_id, file_name, bbox)

    def save_grade(grade_key):
        nonlocal write_header
        split, img_id, file_name, bbox = state["current"]
        row = {
            "split": split,
            "image_id": img_id,
            "file_name": file_name,
            "bbox": bbox,
            "grade": grade_key,
            "grade_label": config.GRADE_LABELS[grade_key],
        }
        append_row(config.LABELS_CSV, row, write_header)
        write_header = False
        state["history"].append(state["idx"])
        state["idx"] += 1
        show_current()

    def go_back():
        if state["history"]:
            state["idx"] = state["history"].pop()
        show_current()

    def skip():
        state["idx"] += 1
        show_current()

    def on_key(event):
        k = event.char.lower()
        if k in config.GRADE_LABELS:
            save_grade(k)
        elif k == "b":
            go_back()
        elif k == "s":
            skip()
        elif k == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    show_current()
    root.mainloop()
    print(f"Saved progress to: {config.LABELS_CSV}")


if __name__ == "__main__":
    main()
