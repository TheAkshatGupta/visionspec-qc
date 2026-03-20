import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from gradcam import get_gradcam_heatmap, overlay_heatmap

# Load model
model = tf.keras.models.load_model("models/pcb_defect_model.h5")

# Image path (change this)
img_path = "dataset/validation/defect/l_light_01_mouse_bite_07_2_600.jpg"

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict
pred = model.predict(img_array)[0][0]
print("Prediction:", "DEFECT" if pred > 0.5 else "PASS")

# Grad-CAM
heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1")

# Overlay
result = overlay_heatmap(heatmap, img_path)

# Save output
cv2.imwrite("outputs/heatmap_result.jpg", result)

print("Grad-CAM result saved at outputs/heatmap_result.jpg")



