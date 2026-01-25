# 📸 FotoFix ML Server

A production‑ready **FastAPI‑based ML server** that combines:

* 🔍 **YOLO object detection**
* 🧠 **OCR text extraction (Tesseract)**
* ✂️ **Background removal (rembg)**

Built for image‑centric workflows such as document processing, photo cleanup, and smart image analysis.

---

## 🚀 Features

* YOLOv8 object detection (CPU‑optimized)
* OCR using Tesseract (pytesseract)
* High‑quality background removal
* Base64 encoded image responses
* Robust error handling & validation
* CORS enabled for frontend usage

---

## 🛠️ Tech Stack

* **FastAPI**
* **Ultralytics YOLO**
* **Tesseract OCR**
* **pytesseract**
* **rembg**
* **Pillow (PIL)**
* **Uvicorn**

---

## 📁 Project Structure

```
phot-fix-ml-fast/
├── src/
│   ├── server.py
│   ├── utils.py
│   ├── uploads/
│   ├── outputs/
│   └── best11.pt
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## ⚙️ Local Setup (Ubuntu)

### 1️⃣ System Dependencies

Install **Tesseract OCR**:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

📍 Default binary path:

```
/usr/bin/tesseract
```

This is already configured in the code:

```python
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
```

---

### 2️⃣ Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3️⃣ Run the Server

```bash
uvicorn src.server:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Interactive docs:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### 🔹 GET /

Service info and available routes.

**Response**

```json
{
  "message": "YOLO + OCR + Background API",
  "routes": {
    "POST /detect": "YOLO object detection",
    "POST /remove-bg": "Remove image background",
    "POST /extract-text": "OCR text extraction"
  }
}
```

---

### 🔹 GET /health

Health check endpoint.

**Response**

```json
{
  "status": "success",
  "message": "Server healthy",
  "code": 200
}
```

---

### 🔹 POST /detect

Run YOLO object detection on an image.

**Request**

* `file` (multipart/form‑data)

**Response**

```json
{
  "boxes": [...],
  "classes": [...],
  "confidences": [...],
  "encoded_image": "<base64>"
}
```

---

### 🔹 POST /remove-bg

Remove background from an image.

**Request**

* `file` (multipart/form‑data)

**Response**

```json
{
  "encoded_image": "<base64>"
}
```

---

### 🔹 POST /extract-text

Extract text using OCR.

**Request**

* `file` (multipart/form‑data)

**Response**

```json
{
  "extracted_text": "Detected text from image"
}
```

---

## 🧪 Notes

* All inference runs on **CPU** by default
* Temporary files are auto‑cleaned after processing
* Base64 responses are frontend‑friendly

---

## 📦 Deployment

* Works on **AWS EC2 / VPS / Docker**
* No GPU required
* Expose using:

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8500
```

---

## 🧠 Use Cases

* Document OCR pipelines
* ID / card scanning
* Photo background removal
* Object detection microservices

---

## 👨‍💻 Author

Built by **Anurag Gupta** 🚀

If you want a **Docker‑only README**, **Swagger examples**, or **frontend integration guide**, just say the word.
