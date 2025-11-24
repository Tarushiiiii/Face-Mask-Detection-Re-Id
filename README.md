# FACE MASK DETECTION + PERSON RE-IDENTIFICATION

### **A Dual-Module Computer Vision System**

This project contains _two independent deep learning pipelines_:

1. **Face Mask Detection**
2. **Person Re-Identification (Re-ID)**

Both systems are modular, standalone, and can be executed separately.

---

# 1. Face Mask Detection

A deep-learning system that detects faces from images/video streams and classifies whether the person is **wearing a mask** or **not wearing a mask**.

## Features

- Face detection using OpenCV
- Mask vs No Mask classifier
- Real-time video & webcam support
- Script for single-image prediction
- Trained `mask_detector.model`

## Face Mask Module Structure

```
face_mask/
│
├── face_detector/
├── train_mask_detector.py
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
└── plot.png
```

## Running Face Mask Detection

### **Image-Based Detection**

```bash
python face_mask/detect_mask_image.py --image examples/test.jpg
```

### **Real-Time webcam**

```bash
python face_mask/detect_mask_video.py
```

# 2. Person Re-Identification (Re-ID)

A deep learning pipeline trained on the **Market-1501 dataset**.
The system extracts **unique person embeddings** and evaluates performance using mAP, Rank-1, Rank-5, and Rank-10 metrics.

## Features

- ResNet-50 based Re-ID architecture
- Market1501 dataset loader
- Embedding extraction
- mAP + CMC evaluation
- Full training loop
- Inference support for new images

## Re-ID Module Structure

```
reid/
├── gallery/
├── output/
├── config.py
├── dataset.py
├── demo_reid.py
├── evaluate.py
├── inference.py
├── manual_match.py
├── match.py
├── model.py
├── train.py
├── trainer.py
└── utils.py
```

## Running Re-ID Training

```bash
python reid/train.py
```

## Running Re-ID Inference

```bash
python reid/inference.py --image examples/person.jpg
```

# Project Structure
```
FACE-MASK-DETECTION-RE-ID/
│
├── dataset/
│   ├── face-mask/
│   └── re-id/
│
├── examples/
├── output/
│
├── src/
│   ├── face_mask/
│   │   ├── face_detector/
│   │   ├── detect_mask_image.py
│   │   ├── detect_mask_video.py
│   │   ├── mask_detector.model
│   │   └── train_mask_detector.py
│   │
│   ├── reid/
│       ├── gallery/
│       ├── output/
│       ├── config.py
│       ├── dataset.py
│       ├── demo_reid.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── manual_match.py
│       ├── match.py
│       ├── model.py
│       ├── train.py
│       ├── trainer.py
│       └── utils.py
│
├── venv/
├── run.py
├── requirements.txt
└── README.md
```
# Installation

## 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Running the Project Using run.py

### A. To run Face Mask Detection Pipeline

```bash
python run.py --task mask
```

### B. To run Person Re-Identification Pipeline

```bash
python run.py --task reid
```

# Future Improvements

- Mobile app dashboard for alerts
- Integration with cloud-based monitoring
- Use YOLOv8 for faster detection
- Multi-camera cross-location Re-ID
- Adding audio alerts and entry control automation

