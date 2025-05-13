from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
from PIL import Image
import io

app = FastAPI()

# Load the trained model
learn = load_learner("export.pkl")
labels = learn.dls.vocab  # List of class labels

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image_bytes = await file.read()
    img = PILImage.create(io.BytesIO(image_bytes))

    # Run prediction
    pred, pred_idx, probs = learn.predict(img)

    # Return prediction and all class probabilities
    return {
        "label": pred,
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))}
    }
