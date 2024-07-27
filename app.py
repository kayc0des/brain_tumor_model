from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from src.preprocessing import DataPreprocessor, DataProcessor
from src.model import Model
from pydantic import BaseModel

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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_path = 'model/braintumor.pkl'
model = Model()  # Instantiate the Model class


class RetrainRequest(BaseModel):
    image_dir: str
    target_dir: str
    target_file: str
    data_dir: str
    epochs: int
    npz_filename: str

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

@app.post('/retrain')
async def retrain(request: RetrainRequest):
    try:
        # Preprocess data
        data_pipeline = DataPreprocessor(
        image_dir=request.image_dir,
        target_dir=request.target_dir,
        target_file=request.target_file,
        data_dir=request.data_dir
        )
        data_pipeline.build_image_as_np()
        data_pipeline.extract_labels()
        data_pipeline.data_to_npz(request.npz_filename)

        # Process data
        data_processor = DataProcessor(os.path.join(request.data_dir, request.npz_filename))
        data_processor.apply_scaler()
        data_processor.apply_one_hot_encoding()
        X_train, X_test, Y_train, Y_test = data_processor.perform_train_test_split()
        data_processor.save_data(train_dir='data/train/', test_dir='data/test/')

        # Save processed data
        data_processor.save_data()

        # Train model
        cnn_model = Model()
        history = cnn_model.train(X_train, Y_train, validation_split=0.2, epochs=request.epochs, batch_size=32)
        
        training_metrics = {
            "loss": history.history['loss'][-1],
            "accuracy": history.history['accuracy'][-1],
            "val_loss": history.history['val_loss'][-1],
            "val_accuracy": history.history['val_accuracy'][-1]
        }

        cnn_model.save_model(filepath='model/braintumor2.pkl')

        return {"status": "Model retraining complete", "metrics": training_metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
