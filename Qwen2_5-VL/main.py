from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.post("/upload")
async def upload(file: UploadFile = File(...), model: str = Form(...)):
    data = await file.read()
    img = Image.open(BytesIO(data))
    w, h = img.size

    return JSONResponse({
        "model": model,
        "filename": file.filename,
        "content_type": file.content_type,
        "width": w, "height": h, "bytes": len(data),
        "result": "a caption"
    })

@app.get("/health")
def health():
    return {"status": "ok"}
