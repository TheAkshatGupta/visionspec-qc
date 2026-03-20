# AI Model Development (Akshat)

## What I did
- Built CNN model for PCB defect classification
- Implemented MobileNetV2 using transfer learning
- Created training pipeline
- Trained model and achieved ~79% accuracy
- Implemented Grad-CAM for defect visualization

## How I did
- Used TensorFlow/Keras for model building
- Used ImageDataGenerator for preprocessing
- Trained model on dataset (pass/defect)
- Applied Grad-CAM to highlight defect regions

## Tools Used
- TensorFlow / Keras
- NumPy
- OpenCV

## Output
- Trained model: pcb_defect_model.h5
- Grad-CAM heatmap output generated

## Files
- cnn_model.py
- mobilenet_model.py
- train_model.py
- gradcam.py
- test_gradcam.py