# 🚗 License Plate Detection & Blurring (Privacy Protection)

## 📌 Overview
This project implements an end-to-end computer vision pipeline to detect and anonymize license plates using YOLOv8. It addresses privacy concerns (GDPR/CCPA) by automatically blurring sensitive information in images.

---

## ⚙️ Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Matplotlib
- Google Colab

---

## 🧠 Approach
1. Data preprocessing and validation
2. Training YOLOv8 for license plate detection
3. Model evaluation using mAP metrics
4. Inference on test images
5. ROI-based Gaussian blurring for anonymization

---

## 📊 Results
- High detection accuracy (mAP@50-95)
- Real-time inference performance (FPS-based evaluation)
- Successful anonymization of license plates
