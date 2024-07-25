from PIL import Image
import numpy as np
import tensorflow as tf
import io

model_path = '../model/braintumor.h5'
model = tf.keras.models.load_model(model_path)

def predict(image):
    # Read image file
    image = Image.open(io.BytesIO(image.read())).convert('RGB')
    image = image.resize((240, 240))
    image_data = np.asarray(image)
    
    if image_data.shape != (240, 240, 3):
        return {"error": "Invalid image shape. Expected (240, 240, 3)."}

    image_data = image_data / 255.0
    image_data = image_data.reshape(1, 240, 240, 3)

    # Perform prediction
    prediction = model.predict(image_data)
    probability = prediction[0][1]

    predicted_class = "Tumor" if probability >= 0.5 else "No Tumor"

    return {
        "prediction": {
            "class": predicted_class,
            "probability": float(probability)
        }
    }
