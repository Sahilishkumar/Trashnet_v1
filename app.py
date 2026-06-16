from flask import Flask, render_template, request
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path="model.tflite")
except ImportError:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path="model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f]

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

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]["index"])[0]

        prediction = labels[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)