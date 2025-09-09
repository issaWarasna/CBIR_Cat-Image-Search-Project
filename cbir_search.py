# import os
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.spatial import distance

# DATASET_DIR = "./cats_set"

# # نفس الدالة لاستخراج histogram
# def extract_features(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (256, 256))
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# # تحميل قاعدة البيانات
# def load_dataset():
#     features = []
#     image_paths = []

#     for img_name in os.listdir(DATASET_DIR):
#         path = os.path.join(DATASET_DIR, img_name)
#         feat = extract_features(path)
#         features.append(feat)
#         image_paths.append(path)
    
#     return features, image_paths

# # البحث عن الصور الأقرب
# def search(query_feature, features, image_paths, top_n=5):
#     dists = []
#     for i, feat in enumerate(features):
#         dist = distance.euclidean(query_feature, feat)
#         dists.append((image_paths[i], dist))
    
#     # ترتيب النتائج من الأقرب للأبعد
#     dists.sort(key=lambda x: x[1])
#     return dists[:top_n]

# # عرض النتائج
# def show_results(results):
#     plt.figure(figsize=(15, 5))
#     for i, (img_path, dist) in enumerate(results):
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.subplot(1, len(results), i + 1)
#         plt.imshow(img)
#         plt.title(f"{dist:.2f}")
#         plt.axis("off")
#     plt.show()

# if __name__ == "__main__":
#     # تحميل الصور
#     features, image_paths = load_dataset()

#     # مسار صورة الاستعلام
#     query_path = "./test1.jpg"  # غيّر هذا حسب اسم الصورة

#     # استخراج الميزات ومطابقة الصور
#     query_feat = extract_features(query_path)
#     results = search(query_feat, features, image_paths, top_n=5)

#     # عرض الصور
#     show_results(results)


import os
import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

# مجلد الصور - غيره حسب مكان الصور عندك
DATASET_DIR = "./cats_set"

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # زيادة عدد البينز لتحسين الدقة
    hist = cv2.calcHist([hsv], [0, 1], None, [90, 100], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_dataset():
    features = []
    image_paths = []
    for img_name in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, img_name)
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            image_paths.append(path)
    return features, image_paths

def compare_histograms(h1, h2, method='correlation'):
    if method == 'euclidean':
        return distance.euclidean(h1, h2)
    elif method == 'chi-square':
        return cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CHISQR)
    elif method == 'correlation':
        # لجعل Correlation يترتب تصاعدي كما في المسافات نضع سالب القيمة
        return -cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL)
    else:
        return distance.euclidean(h1, h2)

def search(query_feature, features, image_paths, top_n=5, method='correlation'):
    dists = []
    for i, feat in enumerate(features):
        dist = compare_histograms(query_feature, feat, method)
        dists.append((image_paths[i], dist))
    dists.sort(key=lambda x: x[1])
    return dists[:top_n]

def show_results(results, query_path):
    plt.figure(figsize=(15, 5))

    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    for i, (img_path, dist) in enumerate(results):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(results) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"{dist:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cbir_search.py path_to_query_image [top_n]")
        sys.exit(1)

    query_path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    features, image_paths = load_dataset()
    query_feat = extract_features(query_path)

    if query_feat is None:
        print(f"Error reading query image: {query_path}")
        sys.exit(1)

    results = search(query_feat, features, image_paths, top_n=top_n, method='correlation')
    show_results(results, query_path)
