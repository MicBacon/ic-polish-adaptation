from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch

model_id = "Qwen/Qwen2-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", dtype="auto"
)

img = Image.open("../shared/data/flickr30k/images/36979.jpg").convert("RGB")

system = "Jesteś ekspertem od opisu obrazów. Pisz po polsku, jasno i bez halucynacji."
user   = "Opisz ten obraz w 3-4 zdaniach. Uwzględnij obiekty, relacje i tło. Nie zgaduj."

messages = [
    {"role": "system", "content": [{"type": "text", "text": system}]},
    {"role": "user",   "content": [
        {"type": "text", "text": user},
        {"type": "image"}
    ]},
]

chat_text = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

inputs = processor(
    text=[chat_text],
    images=[img],
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.05,
    )

resp = processor.batch_decode(out, skip_special_tokens=True)[0]
caption = resp.split("assistant:")[-1].strip()
print(">>>", caption)