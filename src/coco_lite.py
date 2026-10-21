"""
Minimal, pure-Python COCO annotation reader.

Replaces pycocotools for this project so we don't depend on a package
that needs a C compiler to build on Windows. Implements only the
handful of methods our scripts actually use, with the same names and
signatures as pycocotools.coco.COCO, so nothing else in the codebase
needs to change.
"""

import json
from collections import defaultdict


class COCOLite:
    def __init__(self, annotation_file):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.cats = {c["id"]: c for c in data.get("categories", [])}
        self.imgs = {i["id"]: i for i in data.get("images", [])}
        self.anns = {a["id"]: a for a in data.get("annotations", [])}

        self._imgs_by_cat = defaultdict(set)
        self._anns_by_img = defaultdict(list)
        self._anns_by_img_cat = defaultdict(list)
        for a in self.anns.values():
            self._imgs_by_cat[a["category_id"]].add(a["image_id"])
            self._anns_by_img[a["image_id"]].append(a["id"])
            self._anns_by_img_cat[(a["image_id"], a["category_id"])].append(a["id"])

    def getCatIds(self):
        return list(self.cats.keys())

    def loadCats(self, cat_ids):
        return [self.cats[i] for i in cat_ids]

    def getImgIds(self, catIds=None):
        if not catIds:
            return list(self.imgs.keys())
        img_ids = set()
        for cid in catIds:
            img_ids |= self._imgs_by_cat.get(cid, set())
        return sorted(img_ids)

    def getAnnIds(self, imgIds=None, catIds=None):
        imgIds = imgIds or []
        catIds = catIds or []
        ann_ids = []
        for img_id in imgIds:
            if catIds:
                for cid in catIds:
                    ann_ids.extend(self._anns_by_img_cat.get((img_id, cid), []))
            else:
                ann_ids.extend(self._anns_by_img.get(img_id, []))
        return ann_ids

    def loadAnns(self, ann_ids):
        return [self.anns[i] for i in ann_ids]

    def loadImgs(self, img_ids):
        return [self.imgs[i] for i in img_ids]


def COCO(annotation_file):
    """Drop-in name match for pycocotools.coco.COCO so imports read the same."""
    return COCOLite(annotation_file)
