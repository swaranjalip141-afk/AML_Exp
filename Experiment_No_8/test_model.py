import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model(
    "video_action_model.h5"
)

# Class labels
classes = [
    'boxing',
    'handclapping',
    'handwaving',
    'jogging',
    'running',
    'walking'
]

# Load image
img = cv2.imread("test.jpg")

img = cv2.resize(img, (128,128))

img = img / 255.0

img = np.expand_dims(img, axis=0)

# Prediction
prediction = model.predict(img)

class_index = np.argmax(prediction)

print("Predicted Class:",
      classes[class_index])