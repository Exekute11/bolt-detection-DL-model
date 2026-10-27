r"""
Prepare one or more bolt datasets (any common format) for YOLO training.

  python tools/prepare_dataset.py --src "C:\Downloads\bolt-dataset" --own "C:\MyCameraImages" --out dataset --single-class bolt

--src  : public dataset folder(s) (e.g. from Kaggle). Can be given several times.
--own  : YOUR OWN camera images + labels (same format). These are put into the validation set too, so the
         accuracy you see is measured on YOUR camera, not only on Kaggle pictures.
--single-class NAME : merge every class into one class (recommended if you only need "bolt").

Understands automatically: YOLO .txt labels, Pascal-VOC .xml, COCO .json.
Images WITHOUT any label file are skipped (an unlabeled bolt teaches the model to ignore bolts) unless you add
--keep-unlabeled-as-negatives (only do that for pictures that really contain NO bolts).
Output: <out>/images/{train,val}, <out>/labels/{train,val}, <out>/data.yaml  -> use data.yaml for training.
"""
import argparse, json, os, random, shutil, statistics, sys
import xml.etree.ElementTree as ET
import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ap = argparse.ArgumentParser()
ap.add_argument("--src", action="append", default=[], help="public dataset folder")
ap.add_argument("--own", action="append", default=[], help="your own labelled camera images")
ap.add_argument("--out", default="dataset")
ap.add_argument("--single-class", default=None, help="merge all classes into one, e.g. bolt")
ap.add_argument("--val", type=float, default=0.10, help="validation share of public data")
ap.add_argument("--val-own", type=float, default=0.30, help="validation share of your own data")
ap.add_argument("--keep-unlabeled-as-negatives", action="store_true")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
if not a.src and not a.own:
    sys.exit("Give at least one --src or --own folder")
random.seed(a.seed)

names = []                                    # global class names
def cid(name):
    name = name.strip().lower()
    if a.single_class:
        name = a.single_class.lower()
    if name not in names:
        names.append(name)
    return names.index(name)

def read_names(root):
    """class names from data.yaml / classes.txt of a YOLO dataset (list), else None"""
    for f in ("data.yaml", "dataset.yaml", "classes.txt", "obj.names"):
        for dp, _, fs in os.walk(root):
            if f in fs:
                p = os.path.join(dp, f)
                try:
                    if f.endswith(".yaml"):
                        import yaml
                        n = yaml.safe_load(open(p, encoding="utf-8")).get("names")
                        return list(n.values()) if isinstance(n, dict) else list(n)
                    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
                except Exception:
                    pass
    return None

def find_label(img, root):
    stem, d = os.path.splitext(os.path.basename(img))[0], os.path.dirname(img)
    cands = [os.path.join(d, stem + ".txt"), os.path.join(d, stem + ".xml"),
             os.path.join(d.replace("images", "labels"), stem + ".txt"),
             os.path.join(d.replace("images", "Annotations"), stem + ".xml"),
             os.path.join(d.replace("JPEGImages", "Annotations"), stem + ".xml")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None

def load_coco(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json"):
                try:
                    j = json.load(open(os.path.join(dp, f), encoding="utf-8"))
                    if "images" in j and "annotations" in j:
                        cats = {c["id"]: c["name"] for c in j.get("categories", [])}
                        by = {}
                        for an in j["annotations"]:
                            by.setdefault(an["image_id"], []).append(an)
                        for im in j["images"]:
                            idx[os.path.basename(im["file_name"])] = (im["width"], im["height"], by.get(im["id"], []), cats)
                except Exception:
                    pass
    return idx

def parse_yolo(path, src_names):
    out = []
    for l in open(path, encoding="utf-8").read().splitlines():
        p = l.split()
        if len(p) < 5:
            continue
        try:
            k = int(float(p[0])); cx, cy, w, h = [float(v) for v in p[1:5]]
        except ValueError:
            continue
        nm = src_names[k] if src_names and k < len(src_names) else f"class{k}"
        out.append((cid(nm), cx, cy, w, h))
    return out

def parse_voc(path, img):
    r = ET.parse(path).getroot()
    sz = r.find("size")
    W, H = (int(float(sz.find("width").text)), int(float(sz.find("height").text))) if sz is not None else (0, 0)
    if not W or not H:
        im = cv2.imread(img); H, W = im.shape[:2]
    out = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = [float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
        out.append((cid(o.find("name").text), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return out

items = []   # (image path, [(cls,cx,cy,w,h)], role)
skipped_unlabeled = bad = 0
for role, roots in (("public", a.src), ("own", a.own)):
    for root in roots:
        src_names, coco = read_names(root), load_coco(root)
        for dp, _, fs in os.walk(root):
            for f in sorted(fs):
                if os.path.splitext(f)[1].lower() not in IMG_EXT:
                    continue
                img = os.path.join(dp, f)
                try:
                    lab = find_label(img, root)
                    if lab and lab.endswith(".txt"):
                        boxes = parse_yolo(lab, src_names)
                    elif lab:
                        boxes = parse_voc(lab, img)
                    elif f in coco:
                        W, H, anns, cats = coco[f]
                        boxes = [(cid(cats.get(an["category_id"], "bolt")),
                                  (an["bbox"][0] + an["bbox"][2] / 2) / W, (an["bbox"][1] + an["bbox"][3] / 2) / H,
                                  an["bbox"][2] / W, an["bbox"][3] / H) for an in anns]
                    elif a.keep_unlabeled_as_negatives:
                        boxes = []
                    else:
                        skipped_unlabeled += 1
                        continue
                    boxes = [(c, min(max(x, 0), 1), min(max(y, 0), 1), min(w, 1), min(h, 1)) for c, x, y, w, h in boxes
                             if w > 0.002 and h > 0.002]
                    items.append((img, boxes, role))
                except Exception as e:
                    bad += 1
if not items:
    sys.exit("No usable labelled images found. Check the folder path / label format.")

# ---- split & write ----
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    shutil.rmtree(os.path.join(a.out, sub), ignore_errors=True); os.makedirs(os.path.join(a.out, sub))
random.shuffle(items)
counts = {"train": 0, "val": 0}; boxes_total = 0; areas = []; per_class = {}
for n, (img, boxes, role) in enumerate(items):
    split = "val" if random.random() < (a.val_own if role == "own" else a.val) else "train"
    name = f"{role[0]}{n:06d}"
    shutil.copy2(img, os.path.join(a.out, "images", split, name + os.path.splitext(img)[1].lower()))
    with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
        f.write("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes))
    counts[split] += 1; boxes_total += len(boxes)
    for c, x, y, w, h in boxes:
        areas.append(w * h * 100); per_class[names[c]] = per_class.get(names[c], 0) + 1
with open(os.path.join(a.out, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(f"path: {os.path.abspath(a.out).replace(chr(92), '/')}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames: {names}\n")

print("\n================ DATASET READY ================")
print(f"Images: {len(items)}  (train {counts['train']} / val {counts['val']})   Boxes: {boxes_total}")
print("Classes:", per_class)
if skipped_unlabeled: print(f"Skipped {skipped_unlabeled} images with NO label file (kept out on purpose)")
if bad: print(f"Skipped {bad} unreadable files")
print(f"Empty-label images (background/negatives): {sum(1 for i in items if not i[1])}")
if areas:
    med = statistics.median(areas)
    print(f"Median bolt size: {med:.2f}% of the image area")
    if med < 0.5: print("  -> bolts are SMALL: train with --imgsz 960 (or 1280)")
if len(names) > 1: print("  -> several classes found; if you only need bolts, re-run with --single-class bolt")
if not a.own: print("  -> TIP: add --own <folder> with images from YOUR camera for much better real-world accuracy")
print(f"\nNext: python train_improved.py --data {os.path.join(a.out, 'data.yaml')} --weights yolov8s.pt --imgsz 960")
