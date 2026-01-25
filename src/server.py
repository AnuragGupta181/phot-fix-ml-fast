# uvicorn src.server:app --reload

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from ultralytics import YOLO
import uvicorn as uv
import pytesseract
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import os
import shutil
import base64
import uuid

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "best11.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
model.to("cpu")

from src.utils import (
    draw_boxes,
    save_jpg,
)

app = FastAPI(
    title="FotoFix ML Server",
    description="YOLO + OCR + Background Removal API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    message: str
    code: int

class ChatRequest(BaseModel):
    user_id: str
    message: str
    
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

@app.get("/", tags=["Info"])
def info():
    return {
        "message": "YOLO + OCR + Background API",
        "routes": {
            "POST /detect": "YOLO object detection",
            "POST /remove-bg": "Remove image background",
            "POST /extract-text": "OCR text extraction",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="success",
        message="Server healthy",
        code=200,
    )


@app.post("/detect")
async def detect(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"error": "File is required"})
    
    file_path = f"src/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    output_image_data = await draw_boxes(image_path=file_path)

    output_image_path = output_image_data[0]
    boxes, classes, confidences = output_image_data[1], output_image_data[2], output_image_data[3]

    with open(output_image_path, "rb") as f:
        image_bytes = f.read()

    os.remove(file_path)
    shutil.rmtree(os.path.dirname(output_image_path), ignore_errors=True)

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return {"boxes": boxes[0], "classes": classes[0], "confidences": confidences[0], "encoded_image": encoded_image}

@app.post("/remove-bg")
async def remove_bg(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"error": "File is required"})
    
    file_path = f"src/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    uid = uuid.uuid4().hex
    out_name = f"{uid}_processed_bg.jpg"
    output_image_path = os.path.join("src/outputs", out_name)

    img = Image.open(file_path)
    no_bg = remove(img)
    save_jpg(no_bg, output_image_path)

    with open(output_image_path, "rb") as f:
        image_bytes = f.read()

    os.remove(file_path)
    shutil.rmtree(os.path.dirname(output_image_path), ignore_errors=True)

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return {"encoded_image": encoded_image}

@app.post("/extract-text")
async def extract_text(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"error": "File is required"})
    
    file_path = f"src/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)

    os.remove(file_path)

    return {"extracted_text": text}



if __name__ == "__main__":
    uv.run("server:app", host="127.0.0.1", port=8000, reload=True)