"""
Find the best inference settings (confidence / image size) for YOUR video, without retraining.

  python tools/tune_inference.py --weights runs/detect/GodrejVision/Fastener_AI/weights/best.pt \
         --video videos/demo.mp4 --expected 5

--expected = how many bolts really are on the part (optional). The table shows how often each setting
finds exactly that many. Pick the row with the highest 'exact %' and acceptable FPS, then put its
values in settings.json -> "detection": {"conf": ..., "imgsz": ...}
"""
import argparse, time, statistics as st
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--expected", type=int, default=None)
ap.add_argument("--confs", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.35])
ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
ap.add_argument("--frames", type=int, default=60, help="frames to sample from the video")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or a.frames
frames = []
for i in range(a.frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / a.frames))
    ok, f = cap.read()
    if ok:
        frames.append(f)
print(f"{len(frames)} frames loaded\n")

model = YOLO(a.weights)
print(f"{'imgsz':>6} {'conf':>5} | {'avg count':>9} {'median':>6} {'min-max':>8} {'avg conf%':>9} {'exact %':>8} {'FPS':>6}")
for sz in a.imgsz:
    for cf in a.confs:
        counts, confs = [], []
        t0 = time.time()
        for f in frames:
            r = model.predict(f, conf=cf, imgsz=sz, verbose=False)[0]
            counts.append(len(r.boxes))
            confs += [float(b.conf[0]) for b in r.boxes]
        fps = len(frames) / (time.time() - t0)
        exact = f"{100 * sum(c == a.expected for c in counts) / len(counts):6.0f}%" if a.expected else "   n/a"
        print(f"{sz:>6} {cf:>5.2f} | {st.mean(counts):>9.2f} {st.median(counts):>6.0f} "
              f"{min(counts):>3}-{max(counts):<4} {100 * (st.mean(confs) if confs else 0):>9.1f} {exact:>8} {fps:>6.1f}")
print("\nIf even the best row misses bolts -> the MODEL needs better/more training data (see ACCURACY_GUIDE.md).")
