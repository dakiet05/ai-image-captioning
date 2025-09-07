import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

BASE_DIR = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
TRAIN_IMAGES = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
TEST_IMAGES  = os.path.join(BASE_DIR, "Flickr_8k.testImages.txt")
CAPTION_FILE = os.path.join(BASE_DIR, "Flickr8k.token.txt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

def save_tokenizer(tokenizer, path=TOKENIZER_PATH):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path=TOKENIZER_PATH):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_data():
    with open(TRAIN_IMAGES, 'r') as f:
        train_images = f.read().splitlines()
    with open(TEST_IMAGES, 'r') as f:
        test_images = f.read().splitlines()

    with open(CAPTION_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    captions = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        image_name, caption = parts[0].split('#')[0], parts[1].lower()
        caption = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in caption)
        captions.setdefault(image_name, []).append(caption)

    def load_feats(image_list):
        feats = []
        for img in tqdm(image_list, desc="Loading features"):
            p = os.path.join(FEATURE_DIR, img + '.npy')
            if os.path.exists(p):
                feat = np.load(p)
                if feat.shape != (2048,):
                    raise ValueError(f"Feature shape mismatch: {p} has {feat.shape}, expected (2048,)")
                feats.append(feat)
        return np.array(feats)

    train_features = load_feats(train_images)
    test_features = load_feats(test_images)

    all_captions = []
    for img in train_images:
        if img in captions:
            for c in captions[img]:
                all_captions.append(f"startseq {c} endseq")

    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    save_tokenizer(tokenizer)

    def seqs_for(image_list):
        seqs = []
        for img in image_list:
            if img in captions:
                for c in captions[img]:
                    s = f"startseq {c} endseq"
                    seq = tokenizer.texts_to_sequences([s])[0]
                    seqs.append(seq)
        return seqs

    train_sequences = seqs_for(train_images)
    test_sequences  = seqs_for(test_images)

    max_length = max(len(s) for s in (train_sequences + test_sequences))
    print(f"Calculated max_length: {max_length}")
    max_length = min(max_length, 39)
    print(f"Fixed max_length: {max_length}")

    train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')
    test_sequences  = pad_sequences(test_sequences, maxlen=max_length,  padding='post')

    print(f"Shape of train_sequences after padding: {train_sequences.shape}")
    print(f"Shape of test_sequences after padding: {test_sequences.shape}")

    embedding_dim = 256
    embedding_matrix = None  # not used in current model; reserved

    return train_features, test_features, train_sequences, test_sequences, vocab_size, max_length, embedding_dim, embedding_matrix
