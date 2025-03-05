from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
import numpy as np
import cv2
from io import BytesIO
import uvicorn

# Load the trained model
model = keras.models.load_model("brick_classifier.keras")

# Define class names (make sure this matches your training dataset!)
class_names = ["BrickType1", "BrickType2", "BrickType3"]  # <-- Replace with actual brick types

app = FastAPI()

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the brick type from an uploaded image."""
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {"brick_type": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
