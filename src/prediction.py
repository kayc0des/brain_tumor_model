import os
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
MODEL_PATH = '../model/braintumor.pkl'

def predict(image_path):
    """
    Predicts whether an image contains a tumor or not using the pre-trained model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Dictionary containing prediction results and probabilities.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Read and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((240, 240))
    image_data = np.asarray(image)

    if image_data.shape != (240, 240, 3):
        return {"error": "Invalid image shape. Expected (240, 240, 3)."}

    image_data = image_data / 255.0
    image_data = image_data.reshape(1, 240, 240, 3)

    prediction = model.predict(image_data)
    probability = prediction[0][1]

    predicted_class = "Tumor" if probability >= 0.5 else "No Tumor"

    return {
        "prediction": {
            "class": predicted_class,
            "probability": float(probability)
        }
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict tumor presence in an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    result = predict(args.image_path)
    print(result)
