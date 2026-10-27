@echo off
REM ===== One-click training on this laptop (CPU). Edit SRC if your dataset is somewhere else. =====
set SRC=%USERPROFILE%\Downloads\archive
set EPOCHS=50
if not "%~1"=="" set EPOCHS=%~1

echo Dataset folder : %SRC%
echo Epochs         : %EPOCHS%   (run "train_local.bat 3" for a quick test)
echo.
python tools\prepare_dataset.py --src "%SRC%" --out dataset --single-class bolt
if errorlevel 1 goto fail
python train_improved.py --data dataset\data.yaml --weights yolov8n.pt --imgsz 640 --epochs %EPOCHS% --batch 8 --device cpu
if errorlevel 1 goto fail
echo.
echo DONE. Copy the file shown after ">>> NEW MODEL FILE" into runs\detect\GodrejVision\Fastener_AI\weights\
pause
exit /b 0
:fail
echo.
echo Something went wrong - read the message above.
pause
exit /b 1
