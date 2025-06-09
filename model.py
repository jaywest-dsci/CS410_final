import os
import numpy as np
from PIL import Image
from gudhi import CubicalComplex
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def get_persistence_landscape_vector(image_path, num_bins=50):
    img = Image.open(image_path).convert('L').resize((50, 50))  # downsample for speed
    pixels = np.array(img, dtype=np.float64)
    
    cc = CubicalComplex(top_dimensional_cells=-pixels)
    diag = cc.persistence()


    h0 = [pt for pt in diag if pt[0] == 0]
    births_deaths = np.array([pt[1] for pt in h0 if pt[1][1] != np.inf])


    if len(births_deaths) == 0:
        return np.zeros(num_bins)
    lifetimes = births_deaths[:,1] - births_deaths[:,0]
    hist, _ = np.histogram(lifetimes, bins=num_bins, range=(0, np.max(lifetimes)))
    return hist.astype(np.float32)

import matplotlib.pyplot as plt
from gudhi import CubicalComplex

def plot_persistence_diagram(image_path):
    img = Image.open(image_path).convert('L').resize((50, 50))
    pixels = np.array(img, dtype=np.float64)
    cc = CubicalComplex(top_dimensional_cells=-pixels)
    diag = cc.persistence()
    cc.plot_persistence_diagram(diag)
    plt.title("Persistence Diagram (Superlevelset)")
    plt.show()


data_dir = '/Users/jakobrode/Desktop/EC 320/CS410_final/dataset'
X, y = [], []

for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.jpg'):
                path = os.path.join(class_dir, fname)
                vec = get_persistence_landscape_vector(path)
                X.append(vec)
                y.append(label)


le = LabelEncoder()
X = np.array(X)
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print()


print(classification_report(y_test, y_pred, target_names=le.classes_))
