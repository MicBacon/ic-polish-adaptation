import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def _fallback_process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[List[Image.Image]], Optional[List]]:
    images_batch = []
    for m in messages:
        imgs = []
        for c in m.get("content", []):
            if c.get("type") == "image":
                imgs.append(c["image"])
        images_batch.append(imgs if imgs else [Image.new("RGB", (1, 1), color=(0, 0, 0))])
    return images_batch, None

try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
    _process_vision_info = qwen_process_vision_info
except Exception:
    _process_vision_info = _fallback_process_vision_info

def _load_compute_metrics(module_path: Optional[str]):
    if module_path:
        module_path = os.path.abspath(module_path)
        if os.path.isfile(module_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("compute_metrics", module_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            if hasattr(mod, "compute_metrics"):
                return mod.compute_metrics

    def _fallback_metrics(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
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

    return _fallback_metrics


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_image_path(image_id: str, image_root: str, json_dir: str, exts: List[str]) -> Optional[str]:
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
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    num_beams: int = 3,
    temperature: float = 0.0,
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images, videos = _process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            num_beams=num_beams if num_beams and num_beams > 1 and temperature == 0.0 else 1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[:, input_len:]
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return out_text


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja Qwen2.5-VL-7B-Instruct na zbiorze testowym (JSON; PL image captioning).")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id lub ścieżka lokalna.")
    parser.add_argument("--json_path", type=str, required=True, help="Ścieżka do pliku JSON (lista {image_id, captions}).")
    parser.add_argument("--image_root", type=str, default="", help="Katalog bazowy dla obrazów (opcjonalnie).")
    parser.add_argument("--image_exts", type=str, default="jpg,jpeg,png", help="Lista rozszerzeń do sprawdzenia, po przecinkach.")
    parser.add_argument("--compute_metrics_py", type=str, default="", help="Ścieżka do compute_metrics.py (opcjonalne).")
    parser.add_argument("--peft_adapter_path", type=str, default="", help="Ścieżka do adaptera LoRA (opcjonalne).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Urządzenie.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_predictions", type=str, default="predictions.jsonl", help="Plik wyników (predykcje).")
    parser.add_argument("--output_metrics", type=str, default="metrics.json", help="Plik metryk.")

    parser.add_argument("--max_samples", type=int, default=0, help="Jeśli >0, użyj tylko pierwszych N rekordów.")
    parser.add_argument("--sample_index", type=int, default=-1, help="Jeśli >=0, użyj tylko rekordu o tym indeksie (0-based).")

    parser.add_argument("--system_prompt", type=str,
                        default="Jesteś ekspertem od opisu obrazów. Pisz po polsku, jasno i bez halucynacji.")
    parser.add_argument("--user_prompt", type=str,
                        default="Opisz ten obraz w 1 zdaniu. Uwzględnij obiekty, relacje i tło. Nie zgaduj.")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and device.type == "cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if args.peft_adapter_path:
        if not _HAS_PEFT:
            print("[WARN] Podano --peft_adapter_path, ale pakiet 'peft' nie jest dostępny. Ignoruję adapter.")
        else:
            model = PeftModel.from_pretrained(model, args.peft_adapter_path)
            model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    data = read_json(args.json_path)
    json_dir = os.path.dirname(os.path.abspath(args.json_path))

    if args.sample_index >= 0:
        data = data[args.sample_index:args.sample_index + 1]
    elif args.max_samples > 0:
        data = data[:args.max_samples]

    predictions: List[str] = []
    references: List[List[str]] = []
    rows_out: List[Dict[str, Any]] = []
    image_paths_for_metrics: List[str] = []

    exts = [x.strip().lstrip(".").lower() for x in args.image_exts.split(",") if x.strip()]
    total = len(data)
    for idx, item in enumerate(data):
        image_id = str(item.get("image_id", ""))
        refs = [str(x).strip() for x in (item.get("captions") or []) if isinstance(x, str)]
        if not image_id:
            print(f"[{idx}] Brak image_id — pomijam.")
            continue

        img_path = _find_image_path(image_id=image_id, image_root=args.image_root, json_dir=json_dir, exts=exts)
        if not img_path:
            print(f"[{idx}] Nie znaleziono obrazu dla image_id={image_id} (szukano w {args.image_root or json_dir}).")
            continue
        image_paths_for_metrics.append(img_path)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Nie można otworzyć obrazu {img_path}: {e}")
            continue

        pred = generate_caption_for_image(
            model=model,
            processor=processor,
            image=image,
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
        )

        predictions.append(pred)
        references.append(refs if refs else [""])
        rows_out.append({
            "id": image_id,
            "image_path": img_path,
            "prediction": pred,
            "references": refs,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"…przetworzono {idx+1}/{total}")

    # zapisz predykcje
    with open(args.output_predictions, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Zapisano predykcje do: {os.path.abspath(args.output_predictions)}")

    # metryki
    metrics_fn = _load_compute_metrics(args.compute_metrics_py if args.compute_metrics_py else None)
    metrics = {}
    if any(len(r) > 0 and any(x.strip() for x in r) for r in references):
        metrics = metrics_fn(
            predictions, references,
            image_paths=image_paths_for_metrics
        )
    else:
        metrics = {"note": "Brak referencji w pliku testowym - metryki pominięte.", "N": len(predictions)}

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Zapisano metryki do: {os.path.abspath(args.output_metrics)}")
    print("== Wyniki skrót ==")
    for k, v in metrics.items():
        try:
            print(f"{k}: {float(v):.3f}")
        except Exception:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
