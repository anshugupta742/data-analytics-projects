Driver Drowsiness Detection

A computer-vision pipeline that classifies driver face images as drowsy or non-drowsy, built with transfer learning (MobileNetV2) and packaged for edge deployment via TensorFlow Lite.

Overview

The notebook covers the full workflow end to end:

Data loading – mounts the dataset (drowsy / non-drowsy face images, pre-split into train/test) and inspects class balance and image quality.
Face detection & cropping – uses Dlib's frontal face detector to crop each image to the face region (with a fallback to the original image when no face is detected), so the model learns from the eyes/mouth/head pose rather than background clutter.
Preprocessing & augmentation – resizes to 128×128, applies random flips, rotation, zoom, brightness and contrast jitter to simulate day/night and camera variation.
Model training – transfer learning on MobileNetV2 (ImageNet weights), trained in two phases:
Phase 1: frozen base, train a new classification head (15 epochs, lr = 1e-3)
Phase 2: fine-tune the top layers (unfreezing from layer 100 onward) at a low learning rate (15 epochs, lr = 1e-5)
Class weights are used to guard against class imbalance.
Evaluation – confusion matrix, ROC/PR curves, and a threshold sweep to pick an operating point that guarantees ≥ 95% recall on the drowsy class (safety-first, since missing a drowsy driver is far costlier than a false alarm).
Edge deployment – exports the trained model to TensorFlow Lite in several variants (float32, float16, dynamic-range int8, full int8) to compare size vs. accuracy vs. latency trade-offs for running on constrained hardware (e.g. an in-vehicle camera or Raspberry Pi).
Decision layer – a DrowsinessMonitor class turns a noisy stream of per-frame predictions into a stable OK / WARNING / ALERT signal using a sliding window, closer to how the model would actually be used in a vehicle.
Business analysis – written answers on modeling trade-offs (why transfer learning, cost of false negatives vs. false positives, robustness to lighting, why detect the face first, model compression, two-phase training, why accuracy alone is misleading, and suggested next steps for a production deployment).

Configuration

Key parameters (all set in a single config cell near the top of the notebook, so nothing needs to be changed deep in the code):

IMG_SIZE = (128, 128), BATCH_SIZE = 32, VAL_SPLIT = 0.15
USE_DLIB_CROP = True, CROP_MARGIN = 0.15
EPOCHS_P1 = 15 (lr 1e-3), EPOCHS_P2 = 15 (lr 1e-5), FINE_TUNE_AT = 100
TARGET_RECALL = 0.95 — used to pick the decision threshold on validation data


Tech stack:

TensorFlow / Keras (MobileNetV2 transfer learning)
TensorFlow Lite (edge deployment / quantization)
Dlib (face detection)
OpenCV, Pillow (image I/O and processing)
scikit-learn (metrics: precision/recall/F1, ROC/PR curves)
pandas, NumPy, Matplotlib (data handling and visualization)
