# How to get your model from 60–70 % to 90 %+

## 1. First, understand the number
The dashboard "AI accuracy" is the **average confidence** of the boxes, not real accuracy. Real accuracy is
**Recall** (how many real bolts are found) and **Precision** (how many detections are real).
Measure it properly:

    python train_improved.py --data dataset/data.yaml --weights <your best.pt> --eval-only

If Recall is low, bolts are being missed. If confidence is low but Recall is high, the model is just unsure.

## 2. What was changed in the app (no retraining needed)
* Boxes without a tracker ID were drawn as "Bolt" but **not counted** -> visible bolts were "ignored". Now every box is counted
  (`count_all_boxes`).
* Confidence threshold lowered to 0.20 (was YOLO default 0.25) and configurable.
* Optional bigger inference size for small bolts.
Edit `settings.json` -> `"detection"` (restart app). Find the best values for your video:

    python tools/tune_inference.py --weights <best.pt> --video videos/demo.mp4 --expected <real bolt count>

## 3. The real fix: better training data (this is what moves 65 % -> 90 %+)
| Problem | Fix |
|---|---|
| Visible bolts not detected | **Label EVERY bolt in every training image.** An unlabeled bolt teaches the model "this is background". |
| Only a few hundred images | 800-1500+ images; cover different lighting, angles, part positions, shadows, oily/dirty bolts |
| Bolts are small in the frame | Train/infer at `imgsz` 960-1280, move camera closer, or crop to the part area |
| Model too small | Start from `yolov8s.pt` or `yolov8m.pt` instead of `yolov8n.pt` |
| Camera images differ from training images | Collect training images from the SAME camera, lens and lighting used on the line |
| Same mistakes again and again | Use the hard-sample collector (below), fix labels, retrain |

## 4. Hard-sample collector (collect the bolts the model misses)
In `settings.json` set `"detection": { "collect_samples": true }` and run the system normally.
Frames with low confidence, missing bolts or unconfirmed boxes are saved to `dataset_candidates/images` with
auto-labels in `dataset_candidates/labels`. Open them in LabelImg / CVAT / Roboflow, **add the missed bolts**, merge into
your dataset, then fine-tune:

    python train_improved.py --data dataset/data.yaml --weights runs/detect/GodrejVision/Fastener_AI/weights/best.pt

Repeat 2-3 rounds. Each round typically gives a large jump.

## 5. Physical setup (often 20 % of the gain)
Fixed camera and distance, sharp focus, no motion blur (fast shutter), even lighting without glare, plain background.

## 6. Target before going to production
Recall >= 95 %, Precision >= 95 %, mAP@50 >= 95 % on images the model has never seen.
