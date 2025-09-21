#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# YOLOv8 – detekcja obiektów
from ultralytics import YOLO  # pip install ultralytics
# EasyOCR – tekst na obrazie
import easyocr  # pip install easyocr

# Multilingual CLIP do scorowania obraz↔tekst
# dwa warianty – wybierz JEDEN i odkomentuj
# 1) sentence-transformers/clip-ViT-B-32-multilingual-v1 (łatwe API)
from sentence_transformers import SentenceTransformer, util  # pip install sentence-transformers

# 2) (alternatywa) laion/CLIP-ViT-B-32-xlm-roberta-base... via open_clip
# import open_clip  # pip install open_clip_torch

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def pick_base_caption(caps: List[str]) -> str:
    return sorted(caps, key=lambda s: len(s), reverse=True)[0].strip()

PROMPT_PL = """Połącz bazowy podpis z poniższymi dowodami wizualnymi w JEDEN naturalny, precyzyjny podpis po polsku.
Zasady:
- Zachowaj sens bazowego podpisu.
- Dodaj TYLKO informacje obecne w dowodach (nie halucynuj).
- Użyj 1 zdania (max 2), 12-30 słów.
- Jeśli OCR wykrył tekst na obrazie i jest istotny, wstaw go dokładnie.

Bazowy podpis:
"{base}"

Obiekty (skrót listy): {objects}
Tekst na obrazie (OCR): {ocr}

Napisz podpis:"""

def fuse_pl(model, tok, base: str, objects: List[str], ocr_tokens: List[str],
            max_new_tokens=64, temperature=0.3, top_p=0.9, device="cuda"):
    obj_txt = ", ".join(objects) if objects else "brak"
    ocr_txt = ", ".join(ocr_tokens) if ocr_tokens else "brak"
    prompt = PROMPT_PL.format(base=base, objects=obj_txt, ocr=ocr_txt)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
        )
    txt = tok.decode(out[0], skip_special_tokens=True).strip()
    # prosty „postcut”: weź tylko część po ostatnim "Napisz podpis:"
    if "Napisz podpis:" in txt:
        txt = txt.split("Napisz podpis:")[-1].strip()
    # 1 zdanie (lub 2). Utnij nadmiarowe linie.
    txt = txt.split("\n")[0].strip().strip('"')
    return txt

def detect_objects(yolo_model, image_path, conf=0.25, topk=12):
    res = yolo_model.predict(image_path, conf=conf, verbose=False)
    if not res: return []
    r = res[0]
    names = r.names
    if r.boxes is None or len(r.boxes)==0: return []
    labels = r.boxes.cls.cpu().numpy()
    confs  = r.boxes.conf.cpu().numpy()
    agg = {}
    for lab, cf in zip(labels, confs):
        name = names.get(int(lab), str(int(lab)))
        agg.setdefault(name, []).append(float(cf))
    scored = [(k, float(np.mean(v))) for k,v in agg.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [k for k,_ in scored[:topk]]

def ocr(reader, image_path, min_prob=0.55, max_len=32, limit=8):
    try:
        res = reader.readtext(image_path)
    except Exception:
        return []
    out = []
    seen = set()
    for _, text, prob in res:
        if not text or prob is None or prob < min_prob:
            continue
        t = str(text).strip()
        if not t or len(t) > max_len: continue
        tl = t.lower()
        if tl in seen: continue
        seen.add(tl)
        out.append(t)
        if len(out) >= limit: break
    return out

def clip_score_multilingual(model_st, image_path, caption_pl: str):
    # sentence-transformers CLIP-multilingual: obraz przez vision tower w środku API
    # tutaj użyjemy prostej kosinusowej podobieństwa między embedami tekstu i obrazu
    img = Image.open(image_path).convert("RGB")
    # w tym modelu SentenceTransformer.encode przyjmuje też obrazy
    emb_img = model_st.encode([img], convert_to_tensor=True, batch_size=1, normalize_embeddings=True)
    emb_txt = model_st.encode([caption_pl], convert_to_tensor=True, normalize_embeddings=True)
    score = float(util.cos_sim(emb_img, emb_txt)[0,0])
    return score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", required=True)
    p.add_argument("--input_json", required=True)   # tylko TRAIN!
    p.add_argument("--output_json", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fused_per_image", type=int, default=2)
    p.add_argument("--yolo_weights", default="yolov8n.pt")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--ocr_langs", default="pl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clip_threshold", type=float, default=0.28)  # startowy próg; skalibruj na próbce
    args = p.parse_args()

    images_root = Path(args.images_root)
    data = load_json(args.input_json)
    if args.limit:
        data = data[:args.limit]

    # 1) Modele pomocnicze
    yolo = YOLO(args.yolo_weights)
    reader = easyocr.Reader([l.strip() for l in args.ocr_langs.split(",") if l.strip()],
                            gpu=("cuda" in args.device))

    # Multilingual CLIP (SentenceTransformers wrapper)
    st_clip = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device=args.device)

    # 2) Polski LLM – Bielik (4-bit, żeby się zmieścił na 1x24–48GB)
    model_id = "speakleash/Bielik-11B-v2.2-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    out = []
    for rec in tqdm(data, desc="FuseCap-PL"):
        img_id = rec.get("image_id")
        caps = rec.get("captions", [])
        if not img_id or not caps:
            continue
        img_path = images_root / img_id
        if not img_path.exists():
            out.append(rec); continue

        base = pick_base_caption(caps)
        objs = detect_objects(yolo, str(img_path), conf=args.yolo_conf, topk=12)
        ocr_tokens = ocr(reader, str(img_path))

        candidates = []
        # wariant 1: pełne dowody
        candidates.append(fuse_pl(model, tok, base, objs, ocr_tokens, device=args.device))
        # wariant 2: fokus na top obiektach, bez OCR (różnorodność)
        if args.fused_per_image >= 2:
            candidates.append(fuse_pl(model, tok, base, objs[:6], [], device=args.device))

        # sanity: długość + dedup
        cand = [c for c in candidates if 8 <= len(c.split()) <= 35]
        cand = list(dict.fromkeys(cand))

        # 3) Filtr jakości: multilingual CLIP (obraz↔PL)
        good = []
        for c in cand:
            try:
                sc = clip_score_multilingual(st_clip, str(img_path), c)
            except Exception:
                sc = 0.0
            if sc >= args.clip_threshold:
                good.append(c)

        new_caps = caps + good  # dopisujemy do istniejących
        out.append({"image_id": img_id, "captions": new_caps})

    save_json(out, args.output_json)
    print(f"[OK] zapisano {len(out)} rekordów -> {args.output_json}")

if __name__ == "__main__":
    main()
