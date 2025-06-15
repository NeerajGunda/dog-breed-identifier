import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("dog_breed_classifier.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Flip dictionary: index -> class name
class_indices = {int(v): k for k, v in class_indices.items()}

# Prediction function
def predict_dog_breed(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]

    if confidence < 0.5:
        return "❌ Not confident. Possibly not a dog.", f"Confidence: {round(confidence * 100, 2)}%"

    raw_label = class_indices[predicted_index]
    clean_label = raw_label.split("-")[-1].replace("_", " ").title()

    return clean_label, f"Confidence: {round(confidence * 100, 2)}%"

# Gradio Interface
demo = gr.Interface(
    fn=predict_dog_breed,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Predicted Breed"),
        gr.Textbox(label="Confidence")
    ],
    title="🐶 Dog Breed Identifier",
    description="Upload a dog image to identify its breed."
)

demo.launch()

