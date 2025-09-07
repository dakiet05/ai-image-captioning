import os, numpy as np
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

for img_name in tqdm(all_images, desc="No-op transform (validate only)"):
    p = os.path.join(feature_dir, img_name + '.npy')
    if os.path.exists(p):
        feat = np.load(p)
        if feat.shape != (2048,):
            raise ValueError(f"{p} -> {feat.shape}, expected (2048,)")
    else:
        print(f"Missing: {p}")

print("Done.")
