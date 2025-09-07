import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

BASE_DIR = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI"
IMAGE_DIR = os.path.join(BASE_DIR, "Flicker8k_Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()  # (2048,)
])
base_model.trainable = False

def extract_feature(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature.squeeze()  # (2048,)

if __name__ == '__main__':
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in tqdm(image_files, desc="Extracting features"):
        image_path = os.path.join(IMAGE_DIR, image_file)
        output_path = os.path.join(OUTPUT_DIR, image_file + '.npy')
        if not os.path.exists(output_path):
            feat = extract_feature(image_path)
            np.save(output_path, feat)
