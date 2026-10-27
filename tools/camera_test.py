"""
Camera doctor: finds WHY the picture is low resolution / low FPS / blurry ("liquid") and what settings fix it.

    python tools/camera_test.py

Close the dashboard (python app.py) first - only one program can use the camera at a time.
Takes about 1-2 minutes. Send me the printed table if you need help.
"""
import os, sys, time
import cv2

BACKENDS = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)] if os.name == "nt" else [("default", 0)]
SIZES = [(640, 480), (1280, 720), (1920, 1080)]
OUT = "camera_test"; os.makedirs(OUT, exist_ok=True)


def fourcc(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "?"


def measure(cap, n=25):
    for _ in range(3):
        cap.read()                                   # warm-up
    t0, got, last = time.time(), 0, None
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            got += 1; last = f
    dt = time.time() - t0
    return (got / dt if dt > 0 else 0), last


def open_cam(idx, flag):
    cap = cv2.VideoCapture(idx, flag) if os.name == "nt" else cv2.VideoCapture(idx)
    return cap if cap.isOpened() else None


print("\n=== STEP 1: which cameras exist, and what they really deliver (MJPG, 30 FPS requested) ===")
print(f"{'cam#':>4} {'driver':>7} {'asked':>10} {'got':>10} {'mode':>5} {'real FPS':>9}")
results = []
for idx in range(5):
    for bname, flag in BACKENDS:
        cap = open_cam(idx, flag)
        if cap is None:
            continue
        for (w, h) in SIZES:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, 30)
            fps, f = measure(cap)
            if f is None:
                continue
            gh, gw = f.shape[:2]
            print(f"{idx:>4} {bname:>7} {w}x{h:<5} {gw}x{gh:<5} {fourcc(cap):>5} {fps:>9.1f}")
            results.append((idx, bname, flag, w, h, gw, gh, fps))
            cv2.imwrite(os.path.join(OUT, f"cam{idx}_{bname}_{gw}x{gh}.jpg"), f)
        cap.release()
if not results:
    sys.exit("\nNo camera could be opened. Close Teams/Zoom/Camera app, re-plug the USB camera and run again.")

good = [r for r in results if r[5] >= 1280 and r[7] >= 20]
best = max(good or results, key=lambda r: (r[5] * r[6] if r[7] >= 20 else 0, r[7]))
idx, bname, flag, w, h, gw, gh, fps = best
print(f"\nBest found: camera #{idx}, driver {bname}, {gw}x{gh} at {fps:.0f} FPS   (sample pictures saved in '{OUT}')")

print("\n=== STEP 2: exposure test on that camera (auto exposure often makes FPS drop and the picture blurry) ===")
cap = open_cam(idx, flag)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, 30)
ae_auto, ae_man = ((0.75, 0.25) if bname == "dshow" else (1, 0)) if os.name == "nt" else (3, 1)
print(f"{'exposure':>10} {'real FPS':>9} {'brightness %':>13}")
rows = []
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_auto)
fps, f = measure(cap, 30)
if f is not None:
    print(f"{'AUTO':>10} {fps:>9.1f} {cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() / 2.55:>13.0f}")
    rows.append(("AUTO", fps, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() / 2.55))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_man)
for e in (-4, -5, -6, -7, -8, -9):
    cap.set(cv2.CAP_PROP_EXPOSURE, e)
    time.sleep(0.4)
    fps, f = measure(cap, 30)
    if f is None:
        continue
    b = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() / 2.55
    print(f"{e:>10} {fps:>9.1f} {b:>13.0f}")
    rows.append((e, fps, b))
cap.release()

print("\n=== WHAT TO DO ===")
a = next((r for r in rows if r[0] == "AUTO"), None)
man = [r for r in rows if r[0] != "AUTO" and r[1] >= 24 and 25 <= r[2] <= 75]
if gw < 1280:
    print("* This camera/driver only gives", f"{gw}x{gh}.", "Use an external USB 3 webcam, or another Camera # / Driver.")
if a and a[1] < 22 and man:
    e = max(man, key=lambda r: r[1])[0]
    print(f"* AUTO exposure makes the camera slow ({a[1]:.0f} FPS) and blurry. In the dashboard press 'Sharp mode' "
          f"(manual exposure) and set exposure to about {e}. Add more light.")
elif a and a[1] >= 22:
    print("* Auto exposure is fine for FPS. If shaking looks 'liquid', turn OFF Windows Studio Effects (see below).")
if not man:
    print("* The room is too dark for a fast shutter - add a bright lamp/LED light on the part. This fixes blur AND FPS.")
print("* Windows: Settings > Bluetooth & devices > Cameras > your camera > turn OFF Automatic framing, Eye contact,")
print("  Background effects. These crop/zoom the picture (low resolution) and warp it when you move ('liquid').")
print("* Connect the camera to a blue USB 3 port directly (no hub). Close Teams, Zoom and the Windows Camera app.")
