# AI-Convergence-Project
고급AI융합프로젝트(캡스톤디자인)

# Development of Abnormal Behavior Detection Model

## 📖 Project Overview
[cite_start]This project aims to develop a deep learning model that detects abnormal behaviors (e.g., theft) in real-time within unmanned stores using CCTV footage[cite: 1, 5]. [cite_start]By overcoming the limitations of traditional post-event analysis, this system enhances security and operational efficiency[cite: 4, 6].

### Key Features
* [cite_start]**Real-time Detection:** Detects abnormal situations instantly using intelligent CCTV systems[cite: 5].
* [cite_start]**Lightweight:** Designed with approximately 1,000 parameters, enabling real-time operation on edge devices[cite: 9].
* [cite_start]**Fairness:** Focuses solely on skeleton-based action recognition, excluding biases related to appearance, race, or gender[cite: 9].

## 🛠 Methodology & Tech Stack

### Process
1.  [cite_start]**Input:** CCTV or video footage[cite: 7].
2.  [cite_start]**Pose Extraction:** Real-time extraction of 17 keypoints per person using **YOLOv8-Pose**[cite: 7].
3.  [cite_start]**Preprocessing:** Construction of 24-frame sliding window sequences for time-series pattern analysis[cite: 7].
4.  [cite_start]**Anomaly Detection:** Utilizing **STG-NF (Spatio-Temporal Graph Normalizing Flow)** to learn normal behavior distributions[cite: 7].
    * [cite_start]**Normal Behavior:** Mapped to the center of the distribution (High Likelihood)[cite: 9].
    * [cite_start]**Abnormal Behavior:** Mapped to the outside of the distribution (High Negative Log-Likelihood)[cite: 9].

### Tech Stack
* [cite_start]**Model Inference:** PyTorch [cite: 7]
* [cite_start]**Image Processing:** OpenCV [cite: 7]
* [cite_start]**Web Dashboard:** Streamlit [cite: 7]

## 📂 Dataset Setup (PoseLift)

This project utilizes the **PoseLift** dataset. The model is trained using **JSON files** containing structured keypoint data, not the raw `.pkl` files.

**[📂 PoseLift Repository Link](https://github.com/TeCSAR-UNCC/PoseLift/tree/main)**

### Instructions
1.  **Download Data:** Download the **JSON format (`Json_files`)** of the PoseLift dataset from the link above.
2.  **Place Data:** Move the downloaded JSON files into the `STG-NF` directory.
    * *Note:* The JSON files are structured by person ID and frame number, making them suitable for anomaly detection models.

## 🚀 Usage

### 1. Training
Install the required dependencies and run the training script. Since the model uses an unsupervised approach, labels are not required.

```bash
python train_eval.py