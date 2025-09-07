import numpy as np
import os
from tqdm import tqdm

BASE_DIR = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI"
feature_dir = os.path.join(BASE_DIR, 'features')
train_image_file = os.path.join(BASE_DIR, 'Flickr_8k.trainImages.txt')
test_image_file  = os.path.join(BASE_DIR, 'Flickr_8k.testImages.txt')

with open(train_image_file, 'r') as f:
    train_images = f.read().splitlines()
with open(test_image_file, 'r') as f:
    test_images = f.read().splitlines()
all_images = train_images + test_images

for img_name in tqdm(all_images, desc="Checking features"):
    path = os.path.join(feature_dir, img_name + '.npy')
    if os.path.exists(path):
        feat = np.load(path)
        if feat.shape != (2048,):
            raise ValueError(f"Expected (2048,), got {feat.shape} at {path}")
    else:
        print(f"Missing feature: {path}")

sample = os.path.join(feature_dir, '3637013_c675de7705.jpg.npy')
if os.path.exists(sample):
    print("Sample shape:", np.load(sample).shape)
