from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import random

app = Flask(__name__)

# Load model sekali saja saat server start
model = tf.keras.models.load_model("fashion_mnist_model.keras")

# Load dataset (untuk ambil random image)
(_, _), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalisasi (HARUS sama seperti training)
test_data_norm = test_data / 255.0

class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

def image_to_base64(img_array):
    """Convert numpy image -> base64"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route("/predict", methods=["GET"])
def predict_random():
    # Ambil random image
    i = random.randint(0, len(test_data_norm)-1)

    image = test_data_norm[i]
    true_label = class_names[test_labels[i]]

    # Predict
    probs = model.predict(image.reshape(1,28,28), verbose=0)
    pred_index = np.argmax(probs)
    confidence = float(np.max(probs))

    pred_label = class_names[pred_index]

    # Convert image ke base64 supaya Flutter bisa tampilkan
    image_base64 = image_to_base64(image)

    # Buat deskripsi yang enak dibaca
    description = (
        f"Model memprediksi bahwa gambar ini adalah '{pred_label}' "
        f"dengan tingkat kepercayaan {confidence*100:.2f}%. "
        f"Label sebenarnya adalah '{true_label}'."
    )

    return jsonify({
        "prediction": pred_label,
        "confidence": confidence,
        "true_label": true_label,
        "description": description,
        "image_base64": image_base64
    })

if __name__ == "__main__":
    app.run(debug=True)
