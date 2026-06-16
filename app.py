from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model once when app starts
model = tf.keras.models.load_model(
    "model.keras",
    compile=False
)

# Load class labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f]

# Change this only if your model uses a different input size
IMG_SIZE = (128,128)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        image = Image.open(file).convert("RGB")
        image = image.resize(IMG_SIZE)

        img = np.array(image, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)

        prediction = labels[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)