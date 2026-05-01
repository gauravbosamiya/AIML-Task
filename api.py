from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib, json, cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile, os

app = FastAPI(title="Design Demand Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    global base_model, xgb_model, rate_max

    base_model = tf.keras.models.load_model("saved_model/mobilenet_extractor.h5")
    xgb_model  = joblib.load("saved_model/xgb_demand_model.pkl")
    rate_max   = json.load(open("saved_model/meta.json"))["rate_max"]
    print("Models loaded successfully")

@app.get("/health")
def health():
    return {"status": "ok", "rate_max": rate_max}

@app.post("/predict")
async def predict(image: UploadFile = File(...),rate:  float = Form(...)):
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img.astype(np.float32))

        feat    = base_model.predict(np.expand_dims(img, 0), verbose=0)
        img_vec = feat.mean(axis=(1, 2)).flatten()

        rate_norm = np.array([rate / rate_max])
        X_input   = np.hstack([img_vec, rate_norm]).reshape(1, -1)

        pred_qty     = int(round(max(0, np.expm1(xgb_model.predict(X_input)[0]))))
        est_revenue  = int(pred_qty * rate)

        if pred_qty < 20:
            demand_level = "low"
        elif pred_qty < 60:
            demand_level = "moderate"
        else:
            demand_level = "high"

        return {
            "predicted_qty":   pred_qty,
            "rate":            rate,
            "estimated_revenue": est_revenue,
            "demand_level":    demand_level,
            "rate_warning":    rate > rate_max,
        }

    finally:
        os.unlink(tmp_path)  