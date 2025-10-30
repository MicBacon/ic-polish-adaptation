from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

def fallback_process_vision_info(messages):
    images_batch = []
    for m in messages:
        imgs = []
        for c in m.get("content", []):
            if c.get("type") == "image":
                imgs.append(c["image"])
        if imgs:
            images_batch.append(imgs)
        else:
            images_batch.append([Image.new("RGB", (1, 1), color=(0, 0, 0))])
    return images_batch, None

try:
    from qwen_vl_utils import process_vision_info as process_vision_info
except Exception:
    process_vision_info = fallback_process_vision_info

app = FastAPI()

# ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

def model_inference(image, variant_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    messages = [
        {"role": "system", "content": ["Jesteś ekspertem od opisu obrazów. Pisz po polsku, jasno i bez halucynacji."]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Opisz ten obraz w 1 zdaniu. Uwzględnij obiekty, relacje i tło. Nie zgaduj."}]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample = False,
            temperature = None,
            num_beams = 1,
            pad_token_id=processor.tokenizer.eos_token_id,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[:, input_len:]
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return out_text

@app.post("/generate_caption")
async def generate_caption(file: UploadFile = File(...), model: str = Form(...)):
    data = await file.read()
    img = Image.open(BytesIO(data))
    w, h = img.size

    caption = model_inference(img, model)

    return JSONResponse({
        "model": model,
        "filename": file.filename,
        "content_type": file.content_type,
        "width": w, "height": h, "bytes": len(data),
        "caption": caption
    })

@app.get("/health")
def health():
    return {"status": "ok"}
