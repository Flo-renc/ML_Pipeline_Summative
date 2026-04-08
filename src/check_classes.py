import numpy as np

classes = np.load("models/class_names.npy", allow_pickle=True)
print("Loaded class names:", classes)
print("Number of classes:", len(classes))
