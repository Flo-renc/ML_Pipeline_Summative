
import numpy as np


class_names = np.load("models/class_names.npy", allow_pickle=True)
print("Number of classes:", len(class_names))
print(class_names)

