# 🖼️ AI Image Captioning & Assistive App
🚀 A deep learning project that **generates captions for images** and provides **speech output** to assist visually impaired users.  
Built with **TensorFlow, Transformer, InceptionV3, Flask, and gTTS**.
---
## ✨ Features
- 🔎 **Automatic image captioning** using InceptionV3 (feature extractor) + Transformer Decoder.  
- 🗣️ **Text-to-Speech (TTS)** with gTTS to read captions aloud.  
- 🌐 **Web app** built with Flask for easy interaction.  
- 📊 Trained and evaluated on the **Flickr8k dataset**.
---
## ⚙️ Installation
Clone the repo:
```bash
git clone https://github.com/dakiet05/ai-image-captioning.git
cd ai-image-captioning
Create a virtual environment & install requirements:
python -m venv .venv
.venv\Scripts\activate      # (Windows)
# source .venv/bin/activate # (Linux/Mac)
pip install -r requirements.txt
🏋️ Training (optional)
Download Flickr8k dataset (images + captions).
Extract image features:
python extract_features.py
Train model:
python main.py
→ This will save model_best.keras and tokenizer.pkl.
🌐 Run the App
python app.py
Then open http://127.0.0.1:5000 in your browser.
Upload an image (.jpg/.png).
Get auto-generated caption + audio output.
🎯 Example
Input Image
Generated Caption
"A dog is running through the grass"
Audio Output
Speech generated via gTTS 🎧
📊 Tech Stack
Deep Learning: TensorFlow / Keras
Computer Vision: InceptionV3 (feature extraction)
NLP: Transformer Decoder, Tokenizer, BLEU evaluation
Web: Flask, HTML/CSS
TTS: gTTS
🤝 Authors
Dakiet05 (Project owner)
Academic project: AI Generate Caption for Image
📜 License
This project is for educational and research purposes.
Feel free to fork and improve 🚀
