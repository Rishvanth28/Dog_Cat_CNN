"""Load the saved model and predict Cat or Dog."""

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from model import MODEL_PATH


def load_model():
    """Load the configured model and return the Keras model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Place cat_dog_cnn.keras in the project folder."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def get_input_size(model):
    """Get the height and width expected by the saved model."""
    height = model.input_shape[1]
    width = model.input_shape[2]
    if height is None or width is None:
        raise ValueError("The model does not provide a fixed image input size.")
    return int(height), int(width)


def preprocess_image(uploaded_file, model):
    """Convert an upload to RGB, resize it and normalize its pixels."""
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            image = image.convert("RGB")
            height, width = get_input_size(model)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            image_array = np.asarray(image, dtype="float32") / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(
            "Please upload a valid JPG, JPEG, PNG, BMP, GIF or WEBP image."
        ) from error
    return image_array[np.newaxis, ...]


def predict(model, image_array):
    """Return Dog or Cat."""
    output = model.predict(image_array, verbose=0)
    values = np.asarray(output).reshape(-1)
    if values.size == 0:
        raise ValueError("The model returned no prediction.")

    probability = float(values[0])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("The model returned an invalid sigmoid probability.")

    if probability >= 0.5:
        return "Dog"
    return "Cat"
