import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile 
from PIL import Image, UnidentifiedImageError
from fastapi.middleware.cors import CORSMiddleware
import base64

from backend.predict import build_model, preprocessing, predict, SIZE

MODELS_PATH = "./output/improved"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg"}
MODEL_NAMES = ["unet", "unet++", "deeplabv3"]

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cpu")
    ml["device"] = device

    for model_name in MODEL_NAMES:
        weights_path = os.path.join(MODELS_PATH, model_name, "best_model.pth")
        model = build_model(model_name, 1).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        ml[model_name] = model

    yield          # <-- OUTSIDE the loop: the app runs once all models are loaded
    ml.clear()


app = FastAPI(
    title="Breast and Brain Tumor segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://10.0.0.93:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(ml.get("device"))}

@app.post("/segment")
def segment(file: UploadFile = File(...)):

    # 1. Validate the upload
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported"
        )

    data = file.file.read(MAX_FILE_SIZE + 1)

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large (max 10 MB)"
        )

    try:
        original = Image.open(
            io.BytesIO(data)
        ).convert("L")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )

    # 2. Preprocess image
    image = np.array(original)
    x = preprocessing(image)

    # 3. Predict with all models
    results = {}

    for model_name in MODEL_NAMES:

        mask, _ = predict(
            model=ml[model_name],
            x=x
        )

        # Convert mask to PNG
        mask_image = Image.fromarray(
            mask * 255
        )

        buffer = io.BytesIO()

        mask_image.save(
            buffer,
            format="PNG"
        )

        # Convert PNG bytes to Base64
        mask_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        results[model_name] = mask_base64

    return {
        "filename": file.filename,
        "width": SIZE[0],
        "height": SIZE[1],
        "masks": results
    }