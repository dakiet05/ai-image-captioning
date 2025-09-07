import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from gtts import gTTS
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# ====== CONFIG PATHS ======
BASE_DIR = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI"
MODEL_KERAS_BEST = os.path.join(BASE_DIR, "model_best.keras")
MODEL_H5_BEST    = os.path.join(BASE_DIR, "model_best.h5")
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "model.keras")
MODEL_H5_PATH    = os.path.join(BASE_DIR, "model.h5")
TOKENIZER_PATH   = os.path.join(BASE_DIR, "tokenizer.pkl")

# ====== FLASK ======
app = Flask(__name__)

# ====== CUSTOM LAYER (MUST MATCH model.py) ======
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.mha_layers = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
                           for _ in range(num_layers)]
        self.ffn_layers = [tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ]) for _ in range(num_layers)]
        self.layernorm1_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                                  for _ in range(num_layers)]
        self.layernorm2_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                                  for _ in range(num_layers)]
        self.dropout_layers = [tf.keras.layers.Dropout(rate) for _ in range(num_layers)]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return cfg

    def call(self, inputs, training=None):
        x, enc_output = inputs
        for i in range(self.num_layers):
            attn_output = self.mha_layers[i](query=x, key=enc_output, value=enc_output, training=training)
            attn_output = self.dropout_layers[i](attn_output, training=training)
            out1 = self.layernorm1_layers[i](x + attn_output)
            ffn_output = self.ffn_layers[i](out1, training=training)
            ffn_output = self.dropout_layers[i](ffn_output, training=training)
            x = self.layernorm2_layers[i](out1 + ffn_output)
        return x

# ====== LOAD MODEL & TOKENIZER ======
def load_caption_model():
    # Ưu tiên bản best trước
    if os.path.isfile(MODEL_KERAS_BEST):
        return tf.keras.models.load_model(
            MODEL_KERAS_BEST,
            custom_objects={"TransformerDecoder": TransformerDecoder},
            compile=False
        )
    if os.path.isfile(MODEL_H5_BEST):
        return tf.keras.models.load_model(
            MODEL_H5_BEST,
            custom_objects={"TransformerDecoder": TransformerDecoder},
            compile=False
        )
    # Sau đó tới bản final
    if os.path.isfile(MODEL_KERAS_PATH):
        return tf.keras.models.load_model(
            MODEL_KERAS_PATH,
            custom_objects={"TransformerDecoder": TransformerDecoder},
            compile=False
        )
    if os.path.isfile(MODEL_H5_PATH):
        return tf.keras.models.load_model(
            MODEL_H5_PATH,
            custom_objects={"TransformerDecoder": TransformerDecoder},
            compile=False
        )
    raise FileNotFoundError("No model file found. Run training to create model_best.keras or model.keras/.h5.")

model = load_caption_model()

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

assert 'startseq' in tokenizer.word_index, "❌ tokenizer không có 'startseq'"
assert 'endseq' in tokenizer.word_index, "❌ tokenizer không có 'endseq'"

# ====== APP CONFIG ======
max_length = 39
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# ====== FEATURE EXTRACTOR (2048) ======
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
feature_extractor = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])
base_model.trainable = False

def extract_features(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = feature_extractor.predict(img_array, verbose=0)
    return feature.squeeze()  # (2048,)

# ====== BEAM SEARCH ======
def beam_search_predict(model, feature, tokenizer, max_length, beam_width=5):
    start_token = tokenizer.word_index['startseq']
    end_token = tokenizer.word_index['endseq']

    sequences = [[start_token]]
    scores = [0.0]
    feature_expanded = np.expand_dims(feature, axis=0)

    for step in range(max_length - 1):
        candidates = []
        for i, seq in enumerate(sequences):
            score = scores[i]
            seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_length-1, padding='post')
            pred = model.predict([feature_expanded, seq_padded], verbose=0)
            timestep = min(step, pred.shape[1] - 1)
            probs = pred[0, timestep, :]
            top_words = np.argsort(probs)[-beam_width:]
            for w in top_words:
                new_seq = seq + [w]
                new_score = score + float(np.log(probs[w] + 1e-10))
                candidates.append((new_seq, new_score))
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        sequences = [s for s, _ in ordered]
        scores = [sc for _, sc in ordered]
        if any(s[-1] == end_token for s in sequences):
            break

    best_seq = sequences[0]
    inv = tokenizer.index_word
    words = [inv.get(w, '') for w in best_seq if w in inv and w not in (start_token, end_token)]
    return ' '.join([w for w in words if w]).strip()

# ====== TTS ======
def generate_audio(caption, audio_path):
    try:
        tts = gTTS(text=caption, lang='en')
        tts.save(audio_path)
        return True
    except Exception as e:
        print(f"Error generating audio: {e}")
        return False

# ====== ROUTE ======
@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    caption = None
    audio_path = None
    error = None

    if request.method == 'POST':
        image = request.files.get('image')
        if not image or image.filename == '' or not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            error = "Please upload a valid image (JPG/PNG)."
            return render_template('index.html', error=error)

        try:
            image_save_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_save_path)

            features = extract_features(image_save_path)
            caption = beam_search_predict(model, features, tokenizer, max_length, beam_width=10)
            caption = caption.strip()

            if not caption:
                error = "Caption generation failed."
                return render_template('index.html', error=error)

            audio_filename = f"{os.path.splitext(image.filename)[0]}.mp3"
            audio_save_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            if generate_audio(caption, audio_save_path):
                audio_path = '/' + os.path.relpath(audio_save_path, BASE_DIR).replace('\\', '/')
            else:
                error = "Failed to generate audio."

            image_path = '/' + os.path.relpath(image_save_path, BASE_DIR).replace('\\', '/')

        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html', image_path=image_path, caption=caption, audio_path=audio_path, error=error)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
