import os
import cv2
import numpy as np

IMG_SIZE = 150

def load_data(data_dir):
    categories = os.listdir(data_dir)
    data = []

    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(data_dir, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append([img, class_num])
            except Exception as e:
                pass

    return data
