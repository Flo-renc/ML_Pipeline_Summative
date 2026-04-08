from preprocessing import DataPreprocessor
from pathlib import Path

pre = DataPreprocessor(img_size=(32, 32))

print("\n----- 1. TESR LOAD IMAGES FROM FOLDER -----")
X, y, class_names = pre.load_images_from_folder("data/train")

print('Images shape:', X.shape)
print("Labels array shape:", y.shape)
print("Classes found:", class_names)

print("\n----- 2. TEST preprocess_iamges -----")
X_processed = pre.preprocess_images(X)
print("Preprocessed image range:", X_processed.min(), "to", X_processed.max())
print("Processed shape:", X_processed.shape)

print("\n----- 3. TEST validation split -----")
X_train, X_val, y_train, y_val = pre.create_train_val_split(X_processed, y, val_size=0.15)
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)

print("\n----- 4. TEST DATA AUGMENTER -----")
datagen = pre.get_data_augmenter()
sample = X_train[0].reshape(1, 32, 32, 1)
augmented_img = next(datagen.flow(sample, batch_size=1))[0]
print("Augmented image shape:", augmented_img.shape)


print("\n----- 5. TEST create_tf_dataset() -----")
tf_dataset = pre.create_tf_dataset(X_train, y_train, batch_size=32)
print("TF Dataset:", tf_dataset)

print("\n----- 6. TEST load_single_image -----")
first_class = class_names[0]
class_folder = Path(f"data/train/{first_class}")
file_list = list(class_folder.glob("*"))

if len(file_list) == 0:
    raise ValueError(f"No images found under: data/train/{first_class}")


sample_img_path = str(file_list[0])
img_single = pre.load_single_image(sample_img_path)

print("Loaded from:", sample_img_path)
print("Single image shape:", img_single.shape)