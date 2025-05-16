from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner, PILImage
import pandas as pd
import platform
import pathlib
import io

app = FastAPI()

# Fix WindowsPath issues on Linux
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

# Enable CORS so frontend can connect to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and labels
learn = load_learner("export.pkl")
labels = learn.dls.vocab

# Load product recommendation data
df = pd.read_excel("recommendation.xlsx")
df['class'] = df['class'].str.strip().str.lower()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes and convert to PIL
    image_bytes = await file.read()
    img = PILImage.create(io.BytesIO(image_bytes))
    
    # Get prediction
    pred, pred_idx, probs = learn.predict(img)
    top_class = labels[pred_idx].strip().lower()
    class_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # Find product recommendations
    df_temp = df[df['class'] == top_class]
    recommendations = []
    for _, row in df_temp.iterrows():
        recommendations.append({
            "name": row['product_name'],
            "image": row['product_image'],
            "description": row.get('description', ''),
            "price": row.get('price', '0'),
            "benefits": row.get('benefits', '').split(',') if pd.notna(row.get('benefits')) else [],
            "profit_link": row['profit_link']
        })

    return {
        "prediction": pred,
        "probabilities": class_probs,
        "recommendations": recommendations
    }
