import os, sys
import json
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
sys.path.append(os.path.dirname('/workspace/'))

from shared.MetricComputer import MetricComputer

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
JSON_PATH = "../shared/data/flickr30k/flickr30kPolish_captions_test_std.json"
IMAGE_ROOT = "../shared/data/flickr30k/Images"
IMAGE_EXTS = "jpg,jpeg,png"
PEFT_ADAPTER_PATH = ""
DEVICE = "auto"
MAX_NEW_TOKENS = 64
NUM_BEAMS = 3
TEMPERATURE = 0.0
OUTPUT_PREDICTIONS = "predictions.jsonl"
OUTPUT_METRICS = "metrics.json"
MAX_SAMPLES = 0
SAMPLE_INDEX = -1
SYSTEM_PROMPT = "Jesteś ekspertem od opisu obrazów. Pisz po polsku, jasno i bez halucynacji."
USER_PROMPT = "Opisz ten obraz w 1 zdaniu. Uwzględnij obiekty, relacje i tło. Nie zgaduj."

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


def load_compute_metrics(module_path):
    if module_path:
        path = os.path.abspath(module_path)
        if os.path.isfile(path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("compute_metrics", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            if hasattr(mod, "compute_metrics"):
                return mod.compute_metrics

    def fallback_metrics(preds, refs, image_paths=None):
        out = {}
        try:
            import sacrebleu
            max_k = max(len(r) for r in refs) if refs else 0
            ref_sets = []
            for k in range(max_k):
                ref_sets.append([(r[k] if k < len(r) else r[-1]) for r in refs])
            bleu = sacrebleu.corpus_bleu(preds, ref_sets, tokenize="intl")
            out["SacreBLEU"] = float(bleu.score)
        except Exception:
            pass
        try:
            from bert_score import score as bert_score
            first_refs = [r[0] if len(r) > 0 else "" for r in refs]
            _, _, F1 = bert_score(preds, first_refs, lang="pl", rescale_with_baseline=True)
            out["BERTScore_F1"] = float(F1.mean().item())
        except Exception:
            pass
        out["Len_pred_tokens_avg"] = sum(len(p.split()) for p in preds) / max(1, len(preds))
        return out

    return fallback_metrics


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image_path(image_id, image_root, json_dir, exts):
    candidates = []
    for ext in exts:
        if image_root:
            candidates.append(os.path.join(image_root, f"{image_id}.{ext}"))
        candidates.append(os.path.join(json_dir, "Images", f"{image_id}.{ext}"))
        candidates.append(os.path.join(json_dir, f"{image_id}.{ext}"))
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    return None


def generate_caption_for_image(
    model,
    processor,
    image,
    system_prompt,
    user_prompt,
    device,
    max_new_tokens=64,
    num_beams=3,
    temperature=0.0,
):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=(temperature if temperature > 0.0 else None),
            num_beams=(num_beams if num_beams and num_beams > 1 and temperature == 0.0 else 1),
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[:, input_len:]
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return out_text


def main():
    if not JSON_PATH:
        return

    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and device.type == "cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )

    model.eval()

    if PEFT_ADAPTER_PATH:
        if HAS_PEFT:
            model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH)
            model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    data = read_json(JSON_PATH)
    json_dir = os.path.dirname(os.path.abspath(JSON_PATH))

    if SAMPLE_INDEX >= 0:
        data = data[SAMPLE_INDEX:SAMPLE_INDEX + 1]
    elif MAX_SAMPLES > 0:
        data = data[:MAX_SAMPLES]

    predictions = []
    references = []
    rows_out = []
    image_paths_for_metrics = []

    exts = [x.strip().lstrip(".").lower() for x in IMAGE_EXTS.split(",") if x.strip()]
    total = len(data)
    
    mc = MetricComputer()

    for idx, item in enumerate(data):
        image_id = str(item.get("image_id", ""))
        refs = [str(x).strip() for x in (item.get("captions") or []) if isinstance(x, str)]
        if not image_id:
            continue

        img_path = find_image_path(image_id=image_id, image_root=IMAGE_ROOT, json_dir=json_dir, exts=exts)
        if not img_path:
            continue

        image_paths_for_metrics.append(img_path)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        pred = generate_caption_for_image(
            model=model,
            processor=processor,
            image=image,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            temperature=TEMPERATURE,
        )

        predictions.append(pred)
        references.append(refs if refs else [""])
        rows_out.append({"id": image_id, "image_path": img_path, "prediction": pred, "references": refs})

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"{idx+1}/{total}")

    with open(OUTPUT_PREDICTIONS, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(os.path.abspath(OUTPUT_PREDICTIONS))

    if any(len(r) > 0 and any(x.strip() for x in r) for r in references):
        metrics = mc.compute_metrics(predictions, references, image_paths_for_metrics)
    else:
        metrics = {"note": "no_refs", "N": len(predictions)}

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(os.path.abspath(OUTPUT_METRICS))
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()