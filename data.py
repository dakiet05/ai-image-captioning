from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from tqdm import tqdm

BASE_DIR = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI"
caption_file = os.path.join(BASE_DIR, "Flickr8k.token.txt")
train_image_file = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
test_image_file  = os.path.join(BASE_DIR, "Flickr_8k.testImages.txt")
feature_dir = os.path.join(BASE_DIR, "features")

def load_captions(caption_file):
    with open(caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    captions_dict = {}
    for line in lines:
        tokens = line.strip().split('\t')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0].split('#')[0], tokens[1].lower()
        caption = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in caption)
        captions_dict.setdefault(image_id, []).append(caption)
    return captions_dict

def prepare_data():
    captions_dict = load_captions(caption_file)
    with open(train_image_file, 'r') as f:
        train_images = f.read().splitlines()
    with open(test_image_file, 'r') as f:
        test_images = f.read().splitlines()

    train_captions = []
    for img in train_images:
        if img in captions_dict:
            for caption in captions_dict[img]:
                train_captions.append(f"startseq {caption} endseq")

    tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
    tokenizer.fit_on_texts(train_captions)
    train_sequences = tokenizer.texts_to_sequences(train_captions)

    max_length = max(len(seq) for seq in train_sequences)
    print(f"Calculated max_length: {max_length}")
    max_length = 39
    print(f"Fixed max_length: {max_length}")

    train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')

    test_captions = []
    for img in test_images:
        if img in captions_dict:
            for caption in captions_dict[img]:
                test_captions.append(f"startseq {caption} endseq")
    test_sequences = Tokenizer(oov_token="<unk>")
    test_sequences = tokenizer.texts_to_sequences(test_captions)
    test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    def load_feats(imgs):
        arr = []
        for img in tqdm(imgs, desc="Loading features"):
            p = os.path.join(feature_dir, img + '.npy')
            if os.path.exists(p):
                feat = np.load(p)
                if feat.shape != (2048,):
                    raise ValueError(f"{p} has {feat.shape}, expected (2048,)")
                arr.append(feat)
        return np.array(arr)

    train_features = load_feats(train_images)
    test_features = load_feats(test_images)

    # glove optional
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 256
    embedding_matrix = None

    return train_features, train_sequences, test_features, test_sequences, tokenizer, max_length, vocab_size, embedding_matrix, embedding_dim

if __name__ == "__main__":
    prepare_data()
