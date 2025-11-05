from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from peft import PeftModel
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

def model_inference(image, variant_name):
    config = OmegaConf.load(f"configs/{variant_name}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.checkpoint.path:
        model = PeftModel.from_pretrained(model, config.checkpoint.path)

    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    messages = [
        {"role": "system",
         "content": config.prompt.system},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": config.prompt.user}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.params.max_new_tokens,
            do_sample = False,
            temperature = None,
            num_beams = 1,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            no_repeat_ngram_size=config.params.no_repeat_ngram_size,
            repetition_penalty=config.params.repetition_penalty
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[:, input_len:]
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return out_text

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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
    return {"status": "ok", "ctn": "gradio-qwen-ctn"}
