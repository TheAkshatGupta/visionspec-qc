 
# 🚧 VisionSpec QC – Visual Quality Control System

AI-powered Computer Vision system for **PCB (Printed Circuit Board) defect detection** using Deep Learning.

---

## 📌 Project Overview

VisionSpec QC is an intelligent quality inspection system designed for **automated PCB defect detection** in manufacturing environments.

The system analyzes PCB images and classifies them into:

* ✅ PASS (No defect)
* ❌ DEFECT (Soldering issue)

Additionally, it uses **Grad-CAM visualization** to highlight the exact defect location on the PCB.

---

## 🎯 Objectives

* Automate PCB inspection using AI
* Achieve high accuracy with limited industrial data
* Provide explainable predictions using heatmaps
* Build a real-time quality control system

---

## ⚙️ Tech Stack

* **TensorFlow / Keras** → Model Training
* **OpenCV** → Image Processing
* **MobileNetV2** → Transfer Learning
* **Grad-CAM** → Explainability
* **Streamlit** → Web Interface

---

## 🧠 Model Approach

1. Basic CNN (from scratch) for understanding
2. Transfer Learning using MobileNetV2
3. Fine-tuning final layers for PCB classification
4. Grad-CAM for defect localization

---

## 🔄 System Workflow

```
Image Input
     ↓
Preprocessing (Resize + Normalize + Augmentation)
     ↓
AI Model (CNN / MobileNetV2)
     ↓
Prediction (PASS / DEFECT)
     ↓
Grad-CAM Heatmap
     ↓
Final Output (Highlighted Defect)
```

---

## 📁 Project Structure

```
visionspec-qc/
│
├── dataset/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── explainability/
│   ├── inference/
│
├── app/
├── models/
├── outputs/
```

---

## 🚀 Features

* PCB defect classification (binary)
* Transfer learning for better accuracy
* Real-time data augmentation
* Explainable AI using Grad-CAM
* Web-based interface for easy testing

---

## 👥 Team

* **Akshat** – AI Model Development
* **Anushka** – Data Processing & Preprocessing
* **Kashak** – Backend & Integration
* **Nishit** – Frontend & Deployment

---

## 📌 Future Improvements

* Real-time video stream processing
* Edge deployment using TensorFlow Lite
* Multi-class defect detection
* Performance optimization

---

## 📜 License

This project is for educational and research purposes.
