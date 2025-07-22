import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/tumor_detector.h5')
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 3) / 255.0

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return classes[class_index]
