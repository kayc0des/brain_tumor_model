from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

class_labels = ['No Tumor', 'Tumor']

# Allowed origins
origins = [
    "https://localhost",
    "https://localhost:8080",
    "https://localhost:3000"
]

# Define allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_path = 'model/braintumor.h5'
model = tf.keras.models.load_model(model_path)


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read image file
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
