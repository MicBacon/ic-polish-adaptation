#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import importlib.util
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ==== opcjonalne: peft (LoRA adapter przy inferencji) ====
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ---- Qwen-VL helper: process_vision_info (fallback gdy brak qwen_vl_utils) ----
def _fallback_process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[List[Image.Image]], Optional[List]]:
    """Zwraca listę list obrazów (1 lista na wiadomość), brak wideo."""
    images_batch = []
    for m in messages:
        imgs = []
        for c in m.get("content", []):
            if c.get("type") == "image":
                imgs.append(c["image"])
        images_batch.append(imgs if imgs else [Image.new("RGB", (1, 1), color=(0, 0, 0))])
    return images_batch, None

try:
    # jeśli masz zainstalowane qwen_vl_utils, użyjemy ich funkcji
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info  # type: ignore
    _process_vision_info = qwen_process_vision_info
except Exception:
    _process_vision_info = _fallback_process_vision_info


# ---- wczytywanie compute_metrics.py (jeśli podano) + fallbacki ----
def _load_compute_metrics(module_path: Optional[str]):
    """
    Oczekiwany podpis:
        compute_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]
    """
    if module_path:
        module_path = os.path.abspath(module_path)
        if os.path.isfile(module_path):
            spec = importlib.util.spec_from_file_location("compute_metrics", module_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "compute_metrics"):
                return mod.compute_metrics  # type: ignore

    # --- fallback: spróbuj sacrebleu + bert_score (jeśli są), w przeciwnym razie prosty BLEU-4 ---
    def _fallback_metrics(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # sacrebleu
        try:
            import sacrebleu
            # sacrebleu oczekuje listy referencji jako listy list-stringów: [refset1, refset2, ...]
            # przetworzymy refs (N x K) -> K list (po jednej na każdą ref-kolumnę), wyrównując brakujące
            max_k = max(len(r) for r in refs) if refs else 0
            ref_sets = []
            for k in range(max_k):
                ref_sets.append([ (r[k] if k < len(r) else r[-1]) for r in refs ])
            bleu = sacrebleu.corpus_bleu(preds, ref_sets)
            out["BLEU"] = float(bleu.score)
        except Exception:
            # bardzo prosty, przybliżony BLEU-4 (bez smoothingu międzyzdaniowego)
            try:
                from collections import Counter
                def ngrams(tokens, n):
                    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                def sent_bleu(p, rlist, n_max=4):
                    import math
                    ptoks = p.split()
                    if not ptoks:
                        return 0.0
                    precisions = []
                    for n in range(1, n_max+1):
                        p_ngrams = Counter(ngrams(ptoks, n))
                        if not p_ngrams:
                            precisions.append(1e-9)
                            continue
                        max_matches = 0
                        for r in rlist:
                            rtoks = r.split()
                            r_ngrams = Counter(ngrams(rtoks, n))
                            matches = sum(min(c, r_ngrams[g]) for g, c in p_ngrams.items())
                            max_matches = max(max_matches, matches)
                        precisions.append((max_matches / max(1, sum(p_ngrams.values()))) or 1e-9)
                    # geom. mean
                    geom = 1.0
                    for pr in precisions:
                        geom *= pr
                    geom = geom ** (1/len(precisions))
                    # BP pomijamy (nie mamy długości ref „najbliższej”)
                    return geom * 100.0
                bleu_scores = [sent_bleu(p, r) for p, r in zip(preds, refs)]
                out["BLEU"] = float(sum(bleu_scores) / max(1, len(bleu_scores)))
            except Exception:
                out["BLEU"] = 0.0

        # BERTScore (multilingual, często dobrze działa dla PL)
        try:
            from bert_score import score as bert_score
            # flatten refs do wyboru najlepszej referencji per przykład na podstawie F1 (przybliżenie)
            # Tu bierzemy tylko pierwszą ref. Jeśli chcesz pełne macro z max po ref, rozbuduj w razie potrzeby.
            first_refs = [r[0] if len(r) > 0 else "" for r in refs]
            P, R, F1 = bert_score(preds, first_refs, lang="pl", rescale_with_baseline=True)
            out["BERTScore_F1"] = float(F1.mean().item() * 100.0)
        except Exception:
            pass

        # długość zdania
        try:
            import numpy as np
            out["Len_pred_tokens_avg"] = float(np.mean([len(p.split()) for p in preds]))  # type: ignore
        except Exception:
            out["Len_pred_tokens_avg"] = sum(len(p.split()) for p in preds) / max(1, len(preds))
        return out

    return _fallback_metrics


# ---- IO: jsonl ----
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            data.append(json.loads(ln))
    return data


def _extract_fields(
    item: Dict[str, Any],
    image_root: str,
    default_img_dir: str
) -> Tuple[str, List[str], str]:
    sid = str(item.get("id") or item.get("image_id") or item.get("imgid") or item.get("filename") or item.get("file_name") or "")

    cand_img = (
        item.get("image") or item.get("image_path") or item.get("filepath") or
        item.get("file_path") or item.get("file") or item.get("filename") or item.get("file_name")
    )
    if not cand_img:
        raise ValueError("Brak pola ze ścieżką do obrazu (np. 'image').")

    if os.path.isabs(cand_img):
        img_path = cand_img
    else:
        img_path = os.path.join(image_root if image_root else default_img_dir, cand_img)
    img_path = os.path.abspath(img_path)

    refs: List[str] = []
    convs = item.get("conversations") or item.get("messages") or item.get("dialog")
    if isinstance(convs, list):
        for c in convs:
            if not isinstance(c, dict):
                continue
            frm = str(c.get("from") or c.get("role") or "").lower()
            if frm == "gpt" or frm == "assistant" or frm == "ai" or frm == "model":
                val = c.get("value") or c.get("text") or c.get("content") or ""
                if isinstance(val, list):
                    parts = []
                    for el in val:
                        if isinstance(el, str):
                            parts.append(el)
                        elif isinstance(el, dict):
                            parts.append(el.get("text") or el.get("value") or "")
                    val = "\n".join(p for p in parts if p)
                if not isinstance(val, str):
                    continue
                if val.startswith("<image>"):
                    val = val.split("\n", 1)[1] if "\n" in val else ""
                val = val.strip()
                if val:
                    refs.append(val)

    refs = [str(r) for r in refs]

    return img_path, refs, sid


def generate_caption_for_image(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    num_beams: int = 3,
    temperature: float = 0.0,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
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

    # wytnij część wejściową
    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[:, input_len:]
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    # sprzątnij ewentualne sufiksy typu: "Assistant:" itp.
    # (często Qwen kończy tokenem im_end, ale skip_special_tokens powinien go zdjąć)
    return out_text


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja Qwen2.5-VL-7B-Instruct na zbiorze testowym (PL image captioning).")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id lub ścieżka lokalna.")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Ścieżka do zbioru testowego .jsonl (np. flickr30kPolish_test.jsonl).")
    parser.add_argument("--image_root", type=str, default="", help="Katalog bazowy dla względnych ścieżek obrazów (jeśli nie ten sam co test_jsonl).")
    parser.add_argument("--compute_metrics_py", type=str, default="", help="Ścieżka do compute_metrics.py (opcjonalne).")
    parser.add_argument("--peft_adapter_path", type=str, default="", help="Ścieżka do adaptera LoRA (opcjonalne).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Urządzenie.")
    parser.add_argument("--batch_size", type=int, default=1, help="(NIEUŻYWANE – generujemy per obraz dla stabilności Qwen-VL).")
    parser.add_argument("--prompt", type=str, default="Opisz obraz jednym zdaniem po polsku.", help="Instrukcja dla modelu.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_predictions", type=str, default="predictions.jsonl", help="Plik wyników (predykcje).")
    parser.add_argument("--output_metrics", type=str, default="metrics.json", help="Plik metryk.")
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # model + processor
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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

    # wczytaj test jsonl
    data = read_jsonl(args.test_jsonl)
    default_img_dir = os.path.dirname(os.path.abspath(args.test_jsonl))

    predictions: List[str] = []
    references: List[List[str]] = []
    rows_out: List[Dict[str, Any]] = []

    # loop po próbkach (per-image generacja)
    for idx, item in enumerate(data):
        try:
            img_path, refs, sid = _extract_fields(item, args.image_root, default_img_dir)
        except Exception as e:
            print(f"[{idx}] Błąd parsowania rekordu: {e}")
            continue

        if not os.path.isfile(img_path):
            print(f"[{idx}] Brak pliku obrazu: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Nie można otworzyć obrazu {img_path}: {e}")
            continue

        pred = generate_caption_for_image(
            model=model,
            processor=processor,
            image=image,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
        )

        predictions.append(pred)
        references.append(refs)
        rows_out.append({
            "id": sid if sid else str(idx),
            "image_path": img_path,
            "prediction": pred,
            "references": refs,
        })

        if (idx + 1) % 50 == 0:
            print(f"…przetworzono {idx+1}/{len(data)}")

    # zapisz predykcje
    with open(args.output_predictions, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Zapisano predykcje do: {os.path.abspath(args.output_predictions)}")

    # metryki
    metrics_fn = _load_compute_metrics(args.compute_metrics_py if args.compute_metrics_py else None)
    metrics = {}
    if any(len(r) > 0 for r in references):
        metrics = metrics_fn(predictions, references)
    else:
        metrics = {"note": "Brak referencji w pliku testowym - metryki pominięte.", "N": len(predictions)}

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Zapisano metryki do: {os.path.abspath(args.output_metrics)}")
    print("== Wyniki skrót ==")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()