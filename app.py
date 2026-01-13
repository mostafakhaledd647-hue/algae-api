from flask import Flask, request, jsonify
import os
import gdown
import tensorflow as tf
from PIL import Image
import numpy as np

# Disable GPU (important for local d.eployment)
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
MODEL_PATH = "best_resnet101v2.keras"
DRIVE_FILE_ID = "1r6JH7HdKDfhqBT3v4gjYqzbPpOjNM34X"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

classes = [
    "Anabaena",
    "Aphanizomenon",
    "Gymnodinium",
    "Karenia",
    "Microcystis",
    "Noctiluca",
    "Nodularia",
    "Nostoc",
    "Oscillatoria",
    "Prorocentrum",
    "Skeletonema",
    "nontoxic"
]

def preprocessing(image):
    try:
        img = Image.open(image).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        return None

@app.route('/')
def index():
    return "Algae_Image_Classification"

@app.route('/predictApi', methods=['POST'])
def predict_api():
    if 'fileup' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['fileup']
    image_arr = preprocessing(image)

    if image_arr is None:
        return jsonify({'error': 'Invalid image file'}), 400

    preds = model.predict(image_arr, verbose=0)
    idx = int(np.argmax(preds))

    return jsonify({
        'prediction': classes[idx],
        'confidence': float(np.max(preds))
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)


