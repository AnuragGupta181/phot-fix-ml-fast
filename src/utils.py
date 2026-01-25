import os
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from typing import List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "best11.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
model.to("cpu")

async def draw_boxes(image_path, save_dir="src/outputs"):
    results = model(
        image_path,
        conf=0.25,
        save=True,
        project=save_dir,
    )

    boxes: List[List[float]] = []
    classes: List[str] = []
    confidences: List[float] = []

    for b in results[0].boxes:
        boxes.append(b.xyxy[0].tolist())
        cls = int(b.cls[0])
        classes.append(model.names[cls])
        confidences.append(float(b.conf[0]))

    r = results[0]
    save_path = Path(r.save_dir)

    images = list(save_path.glob("*.jpg")) + list(save_path.glob("*.png"))
    if not images:
        raise FileNotFoundError("YOLO output image not found")

    return [str(images[0]), boxes, classes, confidences]


def save_jpg(img: Image.Image, path: str):
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        bg.save(path, "JPEG", quality=90)
    else:
        img.convert("RGB").save(path, "JPEG", quality=90)



# ALLOWED = {"jpg", "jpeg", "png"}

# def allowed_file(filename: str) -> bool:
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


# def make_output_folder(base_dir: str, user_id: str) -> str:
#     path = os.path.join(base_dir, str(user_id))
#     os.makedirs(path, exist_ok=True)
#     return path

# def draw_boxes_and_save(image_path, boxes, classes, out_path):
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)

#     try:
#         font = ImageFont.load_default()
#     except Exception:
#         font = None

#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = [int(round(x)) for x in box]
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
#         label = classes[i]
#         draw.text((x1 + 4, y1 + 4), label, fill="red", font=font)

#     save_jpg(img, out_path)
