import os
import numpy as np
import tensorflow as tf
from preprocess import load_data, load_tokenizer
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

BASE_DIR = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI"

# Load data
train_features, test_features, train_sequences, test_sequences, vocab_size, max_length, embedding_dim, embedding_matrix = load_data()
tokenizer = load_tokenizer()

model = create_model(vocab_size, max_length)
model.summary()

# Expand features to match 5 captions per image (Flickr8k)
X1_train = np.repeat(train_features, 5, axis=0)
X2_train, y_train = train_sequences[:, :-1], train_sequences[:, 1:]

X1_test = np.repeat(test_features, 5, axis=0)
X2_test, y_test = test_sequences[:, :-1], test_sequences[:, 1:]

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

ckpt_keras = os.path.join(BASE_DIR, "model_best.keras")
ckpt_h5    = os.path.join(BASE_DIR, "model_best.h5")
mc_keras = ModelCheckpoint(ckpt_keras, monitor='val_loss', save_best_only=True, verbose=1)
mc_h5    = ModelCheckpoint(ckpt_h5,    monitor='val_loss', save_best_only=True, verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit([X1_train, X2_train], y_train,
          validation_data=([X1_test, X2_test], y_test),
          epochs=50, batch_size=64,
          callbacks=[early_stopping, lr_scheduler, mc_keras, mc_h5],
          verbose=1)

# Save final models (Keras 3 compatible)
model.save(os.path.join(BASE_DIR, "model.keras"))   # recommended
model.save(os.path.join(BASE_DIR, "model.h5"))      # fallback

# Optional: export SavedModel for TF Serving/TFLite (not used by app.py)
try:
    model.export(os.path.join(BASE_DIR, "model_tf"))
    print("Exported SavedModel to model_tf/")
except Exception as e:
    print("Skip export SavedModel:", e)

print("Saved model to model_best.keras (best), model.keras (final), model.h5 (final)")
