import io
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

MODEL_PATH = "model.pt"          # TorchScript model (adapt to your own loading code)
IMG_SIZE = 512                   # input size expected by the model
THRESHOLD = 0.5
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/png", "image/jpeg"}

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup (not on every request)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    ml["model"] = model
    ml["device"] = device
    yield
    ml.clear()


app = FastAPI(title="Segmentation API", version="1.0.0", lifespan=lifespan)


def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0           # H, W, C
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
    return tensor.to(ml["device"])


def predict_mask(image: Image.Image) -> np.ndarray:
    x = preprocess(image)
    with torch.inference_mode():
        logits = ml["model"](x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()          # H, W
    return (prob > THRESHOLD).astype(np.uint8)                # 0 or 1


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    return {"status": "ok", "device": ml.get("device")}


# Plain `def` (not `async def`): inference is blocking/CPU-GPU bound,
# so FastAPI runs it in a thread pool and the server stays responsive.
@app.post(
    "/segment",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
def segment(
    file: UploadFile = File(...),
    output: Literal["mask", "overlay"] = Query("mask"),
):
    # 1) Validate the upload
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only PNG and JPEG images are supported")

    data = file.file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 10 MB)")

    try:
        original = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid or corrupted image")

    # 2) Predict, then resize the mask back to the original size
    mask = predict_mask(original)
    mask_img = Image.fromarray(mask * 255).resize(original.size, Image.NEAREST)

    # 3) Build the output image
    if output == "mask":
        result = mask_img
    else:
        red = Image.new("RGB", original.size, (255, 0, 0))
        result = Image.composite(
            Image.blend(original, red, 0.4), original, mask_img
        )

    return Response(content=to_png_bytes(result), media_type="image/png")