from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib, json, cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile, os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Design Demand Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BLOCKED_CATEGORIES = ["shirt", "jeans", "pant", "t-shirt", "tshirt"] 

@app.on_event("startup")
async def load_models():
    global base_model, classify_model, xgb_model, rate_max

    base_model = tf.keras.models.load_model("saved_model_2/mobilenet_extractor.h5")
    xgb_model = joblib.load("saved_model_2/xgb_demand_model.pkl")
    rate_max = json.load(open("saved_model_2/meta.json"))["rate_max"]

    classify_model = tf.keras.applications.MobileNetV2(
        include_top=True, weights='imagenet', input_shape=(224, 224, 3)
    )
    classify_model.trainable = False
    print("Models loaded successfully")

@app.get("/health")
def health():
    return {"status": "ok", "rate_max": rate_max}

@app.post("/predict")
async def predict(image: UploadFile = File(...), rate: float = Form(...), launch_date: str = Form(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP images allowed.")

    if rate <= 0:
        raise HTTPException(status_code=400, detail="Rate must be greater than 0.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        img = cv2.imread(tmp_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_rgb, (224, 224))
        img_cls = preprocess_input(img_cls.astype(np.float32))
        preds = classify_model.predict(np.expand_dims(img_cls, 0), verbose=0)
        labels = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)
        top_labels = [l[1].lower() for l in labels[0]]

        for label in top_labels:
            for blocked in BLOCKED_CATEGORIES:
                if blocked in label:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prediction blocked for uploaded image category"
                    )

        img_feat = cv2.resize(img_rgb, (224, 224))
        img_feat = preprocess_input(img_feat.astype(np.float32))
        feat = base_model.predict(np.expand_dims(img_feat, 0), verbose=0)
        img_vec = feat.mean(axis=(1, 2)).flatten()

        dt = pd.to_datetime(launch_date)
        month_sin = np.sin(2 * np.pi * dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt.month / 12)
        quarter_norm = dt.quarter / 4.0

        meta = np.array([rate / rate_max, month_sin, month_cos, quarter_norm])
        X_input = np.hstack([img_vec, meta]).reshape(1, -1) 

        pred_qty = int(round(max(0, np.expm1(xgb_model.predict(X_input)[0]))))
        est_revenue = int(pred_qty * rate)

        if pred_qty < 20:
            demand_level = "low"
        elif pred_qty < 60:
            demand_level = "moderate"
        else:
            demand_level = "high"

        return {
            "predicted_qty": pred_qty,
            "rate": rate,
            "launch_date": launch_date,
            "estimated_revenue":est_revenue,
            "demand_level": demand_level,
            "rate_warning": rate > rate_max,
            "detected_labels": top_labels,
        }

    finally:
        os.unlink(tmp_path)