from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_binary_model(path):
    return load_model(path)

def load_personality_model(path):
    return load_model(path)

def preprocess_image(image):
    """
    Preprocess image for models expecting grayscale input.
    Adjust size to match your model's input layer.
    """
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((224, 224))  # Adjust to your model's expected size
    image_array = np.array(image) / 255.0  # Normalize to 0-1
    return image_array

def predict_handwriting(image, model):
    img = preprocess_image(image)
    prediction = model.predict(np.expand_dims(img, axis=0))
    return prediction[0][0] > 0.5  # True if handwriting detected

def predict_personality(image, model):
    img = preprocess_image(image)
    import matplotlib.pyplot as plt
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return {trait: float(pred) for trait, pred in zip(traits, prediction)}
