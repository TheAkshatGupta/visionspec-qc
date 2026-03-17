import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mobilenet_model import build_mobilenet_model

# Paths
train_dir = "dataset/train"
val_dir = "dataset/validation"

# Image size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build model
model = build_mobilenet_model()

# Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Save model
model.save("models/pcb_defect_model.h5")

print("Model training complete and saved!")