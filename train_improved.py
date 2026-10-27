"""
Improved training / fine-tuning script for the Godrej fastener model.

Examples
  # fine-tune your CURRENT model on a bigger, corrected dataset (recommended)
  python train_improved.py --data dataset/data.yaml --weights runs/detect/GodrejVision/Fastener_AI/weights/best.pt

  # train a stronger model from scratch at higher resolution (better for small bolts)
  python train_improved.py --data dataset/data.yaml --weights yolov8s.pt --imgsz 960

  # only measure the real accuracy (precision / recall / mAP) of a model
  python train_improved.py --data dataset/data.yaml --weights best.pt --eval-only
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to data.yaml")
ap.add_argument("--weights", default="yolov8s.pt", help="start from: yolov8s.pt / yolov8m.pt / your best.pt")
ap.add_argument("--imgsz", type=int, default=960, help="960 or 1280 for small bolts (640 misses them)")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--batch", type=int, default=-1, help="-1 = auto")
ap.add_argument("--device", default=None, help="0 for GPU, 'cpu' otherwise")
ap.add_argument("--eval-only", action="store_true")
a = ap.parse_args()

model = YOLO(a.weights)
best_path = a.weights

if not a.eval_only:
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        patience=40,            # stop if no improvement for 40 epochs
        cos_lr=True,            # smoother learning-rate decay
        close_mosaic=15,        # last 15 epochs without mosaic -> cleaner convergence
        # augmentation tuned for an industrial, fixed-camera scene
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   # lighting / colour variation (plant lights, shadows)
        degrees=10, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.1,
        project="runs/detect/GodrejVision", name="Fastener_AI_v2", exist_ok=False,
    )
    best_path = str(model.trainer.best)          # exact path of the new best.pt (folder name may differ)
    model = YOLO(best_path)

m = model.val(data=a.data, imgsz=a.imgsz)
print("\n=========== REAL ACCURACY (validation set) ===========")
print(f"Precision : {m.box.mp * 100:5.1f} %   (of detections made, how many were real bolts)")
print(f"Recall    : {m.box.mr * 100:5.1f} %   (of real bolts, how many were found)  <- 'missed bolts'")
print(f"mAP@50    : {m.box.map50 * 100:5.1f} %")
print(f"mAP@50-95 : {m.box.map * 100:5.1f} %")
print("Target for production: Recall >= 95 %, Precision >= 95 %, mAP@50 >= 95 %")
if not a.eval_only:
    print("\n>>> NEW MODEL FILE:", best_path)
    print(">>> Copy it to runs/detect/GodrejVision/Fastener_AI/weights/best.pt  (keep the old one as backup)")
