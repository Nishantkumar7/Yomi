import os, re, shutil, tempfile, numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModel
import torch

Image.MAX_IMAGE_PIXELS = None
TALL_PAGE_HEIGHT = 3000
SPLIT_CHUNK_HEIGHT = 2500

model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cpu().eval()

app = FastAPI()

def split_image(img, chunk_height):
    w, h = img.size
    return [img.crop((0, y, w, min(y + chunk_height, h))) for y in range(0, h, chunk_height)]

def img_to_np(img): return np.array(img.convert("L").convert("RGB"))

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "upload.pdf")
    with open(pdf_path, "wb") as f: f.write(await file.read())

    images = convert_from_path(pdf_path, dpi=300, fmt='png')
    extracted = []

    for idx, img in enumerate(images):
        segments = split_image(img, SPLIT_CHUNK_HEIGHT) if img.height > TALL_PAGE_HEIGHT else [img]
        for part in segments:
            img_np = img_to_np(part)
            with torch.no_grad():
                boxes = model.predict_detections_and_associations([img_np])
                ocr = model.predict_ocr([img_np], [b["texts"] for b in boxes])
            extracted.extend(ocr[0])

    shutil.rmtree(temp_dir)
    return {"text": "\n".join(extracted)}
