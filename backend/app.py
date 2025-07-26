import os
import re
import shutil
import tempfile
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModel
import uvicorn

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://127.0.0.1:5500"] for stricter settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Increase image size limit
Image.MAX_IMAGE_PIXELS = None

# Configuration
TALL_PAGE_HEIGHT = 3000
SPLIT_CHUNK_HEIGHT = 2500

# Load model on CPU
device = torch.device("cpu")
model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).to(device).eval()

# FastAPI app
app = FastAPI()

def split_tall_image(img, chunk_height=2500):
    width, height = img.size
    return [img.crop((0, y, width, min(y + chunk_height, height))) for y in range(0, height, chunk_height)]

def read_image_as_np_array(image):
    return np.array(image.convert("L").convert("RGB"))

@app.post("/ocr")
async def ocr_pipeline(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Invalid file"}, status_code=400)

    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "upload.pdf")

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    try:
        images = convert_from_path(pdf_path, dpi=300, fmt='png')
        full_text = []

        for idx, img in enumerate(images):
            segments = split_tall_image(img, chunk_height=SPLIT_CHUNK_HEIGHT) if img.height > TALL_PAGE_HEIGHT else [img]

            for segment_img in segments:
                img_np = read_image_as_np_array(segment_img)
                with torch.no_grad():
                    results = model.predict_detections_and_associations([img_np])
                    bboxes = [r["texts"] for r in results]
                    ocr_results = model.predict_ocr([img_np], bboxes)
                full_text.extend(ocr_results[0])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        shutil.rmtree(temp_dir)

    return {"text": "\n".join(full_text)}

# Dynamic port for Render or default 10000 for local dev
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
