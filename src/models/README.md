# AI Model Development (Akshat)

## What I did
- Built a CNN model from scratch for PCB defect classification (Pass/Defect)
- Implemented Transfer Learning using MobileNetV2 for better performance
- Created a training pipeline using ImageDataGenerator
- Implemented Grad-CAM for visualizing defect regions

## How I did
- Used TensorFlow/Keras to design and train models
- Loaded pretrained MobileNetV2 and replaced final layers for binary classification
- Built a training script to load images and train the model
- Applied Grad-CAM to highlight important regions in the image

## Tools Used
- TensorFlow / Keras
- NumPy
- OpenCV

## Output
- Model ready for training (pcb_defect_model.h5 will be generated after training)
- Grad-CAM heatmap generation implemented

## Files
- cnn_model.py
- mobilenet_model.py
- train_model.py
- gradcam.py