"""
Make detection 2-4x faster on a CPU-only PC (no retraining, same accuracy).

  pip install openvino
  python tools/export_fast.py runs/detect/GodrejVision/Fastener_AI/weights/best.pt

Then run the app with the exported folder:
  set BOLT_MODEL=runs\detect\GodrejVision\Fastener_AI\weights\best_openvino_model
  python app.py
"""
import sys
from ultralytics import YOLO

w = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
print("Exported to:", YOLO(w).export(format="openvino", imgsz=imgsz, half=False))
