"""
Brain Tumor Detection - FastAPI Application
==========================================
Usage:
    pip install fastapi uvicorn torch torchvision pillow python-multipart
    uvicorn main:app --reload --port 8000

Place brain_tumor_model.pth (downloaded from Colab) in the same directory.
API docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import time
from typing import Optional

# ─── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🧠 Brain Tumor Detection API",
    description="Detects brain tumors from MRI images using ResNet18",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "brain_tumor_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
classes = None
model_info = {}


def load_model():
    global model, classes, model_info
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Auto-detect number of output classes from the checkpoint weights
        state = checkpoint["model_state_dict"]
        num_classes = state["fc.1.weight"].shape[0]  # e.g. 2 or 4
        print(f"🔎 Detected {num_classes} output classes from checkpoint")

        # Rebuild ResNet18 matching the exact architecture used in training
        net = models.resnet18(weights=None)
        num_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        model = net
        # Default 4-class names used in most public brain-tumor datasets
        default_classes = (
            ["no", "yes"] if num_classes == 2
            else ["glioma", "meningioma", "no_tumor", "pituitary"]
        )
        classes = checkpoint.get("classes", default_classes)
        model_info = {
            "architecture": checkpoint.get("architecture", "resnet18"),
            "input_size": checkpoint.get("input_size", 224),
            "val_accuracy": checkpoint.get("val_accuracy", None),
            "classes": classes,
            "device": str(device),
            "num_classes": num_classes,
        }
        print(f"✅ Model loaded on {device} | Classes: {classes}")
    except FileNotFoundError:
        print(f"⚠️  Model file '{MODEL_PATH}' not found. Download it from Colab first.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    load_model()


# ─── Image Preprocessing ───────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image) -> dict:
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    label = classes[pred_idx]
    confidence = probs[pred_idx].item()
    probabilities = {cls: round(probs[i].item(), 4) for i, cls in enumerate(classes)}

    # Tumor present if prediction is NOT the no-tumor class
    has_tumor = label not in ("no", "no_tumor")

    return {
        "prediction": label,
        "has_tumor": has_tumor,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="API Info")
def root():
    return {
        "message": "Brain Tumor Detection API",
        "docs": "/docs",
        "endpoints": {
            "POST /predict": "Upload MRI image for tumor detection",
            "POST /predict/base64": "Send base64-encoded image",
            "GET /health": "Health check",
            "GET /model/info": "Model information",
        }
    }


@app.get("/health", summary="Health Check")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "device": str(device),
        "model_loaded": model is not None,
    }


@app.get("/model/info", summary="Model Info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


@app.post("/predict", summary="Predict from Image Upload")
async def predict_upload(file: UploadFile = File(..., description="MRI brain scan image (JPG/PNG)")):
    """
    Upload an MRI image and get tumor detection results.

    Returns:
    - **prediction**: 'yes' (tumor) or 'no' (no tumor)
    - **has_tumor**: boolean
    - **confidence**: probability of the predicted class
    - **probabilities**: probability for each class
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place brain_tumor_model.pth in the app directory.")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Use JPG or PNG.")

    # Read and process image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {str(e)}")

    start = time.time()
    result = predict(image)
    elapsed = round(time.time() - start, 3)

    return JSONResponse({
        **result,
        "filename": file.filename,
        "inference_time_seconds": elapsed,
        "message": "⚠️ Tumor detected!" if result["has_tumor"] else "✅ No tumor detected",
    })


@app.post("/predict/base64", summary="Predict from Base64 Image")
async def predict_base64(data: dict):
    """
    Send a base64-encoded image for tumor detection.

    Request body:
    ```json
    { "image": "<base64_string>" }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body.")

    try:
        # Handle data URI (data:image/jpeg;base64,...)
        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    start = time.time()
    result = predict(image)
    elapsed = round(time.time() - start, 3)

    return JSONResponse({
        **result,
        "inference_time_seconds": elapsed,
        "message": "⚠️ Tumor detected!" if result["has_tumor"] else "✅ No tumor detected",
    })