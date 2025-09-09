import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# مسار الصور
DATASET_DIR = "./cats_set"

# استخراج histogram (لون HSV)
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# تحميل جميع الصور واستخراج ميزاتها
def load_dataset():
    features = []
    image_paths = []

    for img_name in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, img_name)
        feat = extract_features(path)
        features.append(feat)
        image_paths.append(path)
    
    return features, image_paths

if __name__ == "__main__":
    feats, paths = load_dataset()
    print(f"{len(feats)} is loaded from the dataset.")
