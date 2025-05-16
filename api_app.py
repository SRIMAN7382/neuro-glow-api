from fastapi import FastAPI
import gradio as gr
from fastai.vision.all import *
import pandas as pd
import platform
import pathlib

app = FastAPI()

# Fix for WindowsPath error
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('export.pkl')
labels = learn.dls.vocab
df = pd.read_excel("recommendation.xlsx")
df['class'] = df['class'].str.strip().str.lower()

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    top_class = labels[pred_idx].strip().lower()

    class_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}

    df_temp = df[df['class'] == top_class]
    if df_temp.empty:
        recommended_html = "<p>No product recommendations found for this condition.</p>"
    else:
        recommended_html = "".join([
            f"""
            <div style="text-align:center;margin-bottom:20px;">
                <a href="{row['profit_link']}" target="_blank">
                    <img src="{row['product_image']}" width="150"><br>
                    <strong>{row['product_name']}</strong>
                </a>
            </div>
            """ for _, row in df_temp.iterrows()
        ])

    return class_probs, recommended_html

# Gradio UI
gr_interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[gr.Label(label="Prediction Probabilities"), gr.HTML()],
    title="🧠 Face Skin Analyzer (NeuroGlow AI)",
    description="Upload a facial image to detect skin conditions and get product recommendations."
)

# Mount to FastAPI
app = gr.mount_gradio_app(app, gr_interface, path="/")
