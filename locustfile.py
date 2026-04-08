from locust import HttpUser, task, between
import random
from pathlib import Path
import os

TEST_IMAGES_DIR = Path("data/test")

def get_random_image():
    base_dir = "data/test"
    all_images = []

    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)

        if os.path.isdir(label_path):
            for img in os.listdir(label_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_images.append(os.path.join(label_path, img))

    img_path = random.choice(all_images)

    with open(img_path, "rb") as f:
        return f.read()

class CharacterPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        image_bytes = get_random_image()

        self.client.post(
            "/predict",
            files={"file": ("image.png", image_bytes, "image/png")}
        )