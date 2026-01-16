# ============================
# FotoFix FastAPI Server (FINAL)
# ============================

import os
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from pydantic import BaseModel

from ultralytics import YOLO
from rembg import remove
from PIL import Image
import pytesseract

from src.utils import (
    allowed_file,
    make_output_folder,
    save_jpg,
    draw_boxes_and_save,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "best11.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Force CPU usage (EC2 safe)
model = YOLO(MODEL_PATH)
model.to("cpu")

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="FotoFix API",
    description="YOLO + OCR + Background Removal API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Pydantic Models (Responses)
# --------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    message: str
    code: int


class DetectionBBox(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]


class DetectResponse(BaseModel):
    status: str
    output_url: str
    detections: List[DetectionBBox]
    object_types: List[str]


class SimpleOutputResponse(BaseModel):
    status: str
    output_url: str


class OCRResponse(BaseModel):
    status: str
    extracted_text: str


class FindAllResponse(BaseModel):
    images: List[str]

# --------------------------------------------------
# GLOBAL EXCEPTION HANDLERS
# --------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Invalid request data",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "details": str(exc),
        },
    )

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/", tags=["Info"])
def info():
    return {
        "message": "YOLO + OCR + Background API",
        "routes": {
            "POST /detect": "YOLO object detection",
            "POST /remove-bg": "Remove image background",
            "POST /extract-text": "OCR text extraction",
            "POST /find-all": "List user output images",
            "GET /outputs/{user_id}/{file_name}": "Serve output image",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="success",
        message="Server healthy",
        code=200,
    )

# --------------------------------------------------
# YOLO DETECTION
# --------------------------------------------------
@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(
    image: UploadFile = File(...),
    user_id: str = Form(...),
):
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(400, "Invalid file type")

    uid = uuid.uuid4().hex
    upload_path = os.path.join(UPLOAD_DIR, f"{uid}_{image.filename}")

    with open(upload_path, "wb") as f:
        f.write(await image.read())

    try:
        results = model.predict(source=upload_path, save=False)[0]

        boxes: List[List[float]] = []
        classes: List[str] = []
        confidences: List[float] = []

        for b in results.boxes:
            boxes.append(b.xyxy[0].tolist())
            cls = int(b.cls[0])
            classes.append(model.names[cls])
            confidences.append(float(b.conf[0]))

        out_folder = make_output_folder(OUTPUT_DIR, user_id)
        out_name = f"{uid}_processed_detect.jpg"
        out_path = os.path.join(out_folder, out_name)

        draw_boxes_and_save(upload_path, boxes, classes, out_path)

        detections = [
            DetectionBBox(
                class_name=classes[i],
                confidence=confidences[i],
                bbox=boxes[i],
            )
            for i in range(len(boxes))
        ]

        return DetectResponse(
            status="success",
            output_url=f"/outputs/{user_id}/{out_name}",
            detections=detections,
            object_types=list(set(classes)),
        )

    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

# --------------------------------------------------
# BACKGROUND REMOVAL
# --------------------------------------------------
@app.post("/remove-bg", response_model=SimpleOutputResponse, tags=["Background"])
async def remove_bg(
    image: UploadFile = File(...),
    user_id: str = Form(...),
):
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(400, "Invalid file type")

    uid = uuid.uuid4().hex
    upload_path = os.path.join(UPLOAD_DIR, f"{uid}_{image.filename}")

    with open(upload_path, "wb") as f:
        f.write(await image.read())

    try:
        img = Image.open(upload_path)
        no_bg = remove(img)

        out_folder = make_output_folder(OUTPUT_DIR, user_id)
        out_name = f"{uid}_processed_bg.jpg"
        out_path = os.path.join(out_folder, out_name)

        save_jpg(no_bg, out_path)

        return SimpleOutputResponse(
            status="success",
            output_url=f"/outputs/{user_id}/{out_name}",
        )

    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

# --------------------------------------------------
# OCR EXTRACTION
# --------------------------------------------------
@app.post("/extract-text", response_model=OCRResponse, tags=["OCR"])
async def extract_text(
    image: UploadFile = File(...),
    user_id: str = Form(...),
):
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(400, "Invalid file type")

    uid = uuid.uuid4().hex
    upload_path = os.path.join(UPLOAD_DIR, f"{uid}_{image.filename}")

    with open(upload_path, "wb") as f:
        f.write(await image.read())

    try:
        text = pytesseract.image_to_string(Image.open(upload_path))
        return OCRResponse(
            status="success",
            extracted_text=text.strip(),
        )

    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

# --------------------------------------------------
# FIND ALL USER OUTPUTS
# --------------------------------------------------
@app.post("/find-all", response_model=FindAllResponse, tags=["Outputs"])
def find_all(user_id: str = Form(...)):
    folder = os.path.join(OUTPUT_DIR, user_id)
    if not os.path.exists(folder):
        return FindAllResponse(images=[])

    files = sorted(os.listdir(folder))
    return FindAllResponse(
        images=[f"/outputs/{user_id}/{f}" for f in files]
    )

# --------------------------------------------------
# SERVE OUTPUT FILES
# --------------------------------------------------
@app.get("/outputs/{user_id}/{file_name}", tags=["Outputs"])
def serve_output(user_id: str, file_name: str):
    path = os.path.join(OUTPUT_DIR, user_id, file_name)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")

    return FileResponse(path)

# --------------------------------------------------
# RUN COMMAND (DO NOT USE python server.py)
# --------------------------------------------------
# uvicorn src.server:app --reload
