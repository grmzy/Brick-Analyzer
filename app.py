from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
import numpy as np
import cv2
import uvicorn
import os
import gdown

# Google Drive File ID (Replace with actual ID if needed)
GOOGLE_DRIVE_FILE_ID = "1B7FKnGXRX4RMx_b2w_j-O9oWHbbSQZe-"
MODEL_PATH = "brick_classifier.keras"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id=1B7FKnGXRX4RMx_b2w_j-O9oWHbbSQZe-", MODEL_PATH, quiet=False)

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Define class names (Replace these with actual brick types)
class_names = ["BrickType1", "BrickType2", "BrickType3"]

# Initialize FastAPI app
app = FastAPI(
    title="Brick Analyzer API",
    description="Upload a brick image to classify its type.",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.get("/")
def home():
    """Root endpoint to check if the API is running."""
    return {"message": "Brick Analyzer API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the brick type from an uploaded image."""
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {"brick_type": predicted_class, "confidence": confidence}

# Run API only when executing this script directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
