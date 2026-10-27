# Train a better bolt model - step by step (beginner)

## 0. Important rules
* Kaggle pictures are NOT your camera. Always add **your own** labelled pictures (from the real part / real camera).
  Aim for at least 100-200 of your own images; more is better.
* Every bolt in every picture must have a box. Unlabeled bolts teach the model to ignore bolts.
* Add some pictures with NO bolts (background, hands, faces, phones) with empty labels - this cuts false detections.

## 1. Prepare the data (on your PC)
    pip install -r requirements.txt
    python tools/prepare_dataset.py --src "C:\path\to\kaggle\bolt-dataset" --own "C:\path\to\my_camera_images" --out dataset --single-class bolt
Read the report it prints (images, boxes, bolt size). Output: `dataset/data.yaml`.

## 2A. Train on a free GPU (recommended if your PC has no NVIDIA GPU)
Kaggle: New Notebook -> Settings -> Accelerator **GPU T4** -> upload the `dataset` folder as a Dataset -> run:
    !pip install -q ultralytics
    from ultralytics import YOLO
    import yaml, glob
    p = glob.glob('/kaggle/input/**/data.yaml', recursive=True)[0]            # your uploaded dataset
    cfg = yaml.safe_load(open(p)); cfg['path'] = p.rsplit('/',1)[0]
    yaml.safe_dump(cfg, open('/kaggle/working/data.yaml','w'))
    YOLO('yolov8s.pt').train(data='/kaggle/working/data.yaml', epochs=120, imgsz=960, patience=30, cos_lr=True,
                             close_mosaic=15, degrees=10, scale=0.5, hsv_v=0.4, project='/kaggle/working/runs', name='bolt')
Download `/kaggle/working/runs/bolt/weights/best.pt`. (Google Colab works the same way.)

## 2B. Train on your PC
    python train_improved.py --data dataset/data.yaml --weights yolov8s.pt --imgsz 960
(add `--device 0` with an NVIDIA GPU; on CPU it can take many hours - use 2A instead.)

## 3. Check real accuracy
    python train_improved.py --data dataset/data.yaml --weights runs/detect/GodrejVision/Fastener_AI_v2/weights/best.pt --eval-only
Look at **Recall** (missed bolts) and **Precision** (false detections). Target: both >= 95 %.

## 4. Use the new model
Copy the new `best.pt` to `runs/detect/GodrejVision/Fastener_AI/weights/best.pt` (keep the old one as backup)
and run `python app.py`. Then test with the REAL part. If bolts are still missed, turn on `collect_samples`
(see ACCURACY_GUIDE.md), label those frames, add them with `--own`, and train again. Repeat 2-3 times.

## Easiest way (no Kaggle GPU needed): Google Colab
Open `tools/Train_in_Colab.ipynb` in Google Colab (colab.research.google.com -> File -> Upload notebook),
choose Runtime -> T4 GPU, drag `dataset.zip` into the Files panel, then Runtime -> Run all.

## Train inside VS Code (CPU laptop)
1. VS Code -> File -> Open Folder -> the project folder. Terminal -> New Terminal.
2. `pip install -r requirements.txt`
3. Prepare data (see step 1 above), then a quick test:  
   `python train_improved.py --data dataset/data.yaml --weights yolov8n.pt --imgsz 640 --epochs 3 --batch 8 --device cpu`
4. Full run (leave plugged in, sleep OFF):  
   `python train_improved.py --data dataset/data.yaml --weights yolov8n.pt --imgsz 640 --epochs 50 --batch 8 --device cpu`
5. When finished it prints ">>> NEW MODEL FILE: ..." - copy that best.pt to `runs/detect/GodrejVision/Fastener_AI/weights/`.

## One-click on Windows
`train_local.bat` (dataset expected at Downloads\archive) prepares the data and trains. `train_local.bat 3` = quick 3-epoch test.
