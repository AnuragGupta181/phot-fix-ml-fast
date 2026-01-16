import os
from PIL import Image, ImageDraw, ImageFont

ALLOWED = {"jpg", "jpeg", "png"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def make_output_folder(base_dir: str, user_id: str) -> str:
    path = os.path.join(base_dir, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def save_jpg(img: Image.Image, path: str):
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        bg.save(path, "JPEG", quality=90)
    else:
        img.convert("RGB").save(path, "JPEG", quality=90)


def draw_boxes_and_save(image_path, boxes, classes, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = classes[i]
        draw.text((x1 + 4, y1 + 4), label, fill="red", font=font)

    save_jpg(img, out_path)
