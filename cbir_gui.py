import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
from scipy.spatial import distance

# إعداد بيانات الصور
DATASET_DIR = "./cats_set"

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
    plt.title("Query")
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

# واجهة Tkinter
class CBIRApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CBIR - Image Search Engine")
        self.master.geometry("800x500")

        self.query_path = None
        self.top_n = tk.IntVar(value=5)

        tk.Button(master, text="اختر صورة الاستعلام", command=self.load_query).pack(pady=10)
        tk.Label(master, text="عدد الصور المشابهة:").pack()
        tk.Entry(master, textvariable=self.top_n).pack()

        tk.Button(master, text="ابحث الآن", command=self.run_search).pack(pady=20)

    def load_query(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.query_path = path
            messagebox.showinfo("تم التحميل", f"تم تحميل الصورة:\n{path}")

    def run_search(self):
        if not self.query_path:
            messagebox.showerror("خطأ", "يجب اختيار صورة استعلام أولاً")
            return

        features, image_paths = load_dataset()
        query_feat = extract_features(self.query_path)
        if query_feat is None:
            messagebox.showerror("خطأ", "فشل في قراءة صورة الاستعلام")
            return

        results = search(query_feat, features, image_paths, top_n=self.top_n.get(), method='correlation')
        show_results(results, self.query_path)

# تشغيل الواجهة
if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()
