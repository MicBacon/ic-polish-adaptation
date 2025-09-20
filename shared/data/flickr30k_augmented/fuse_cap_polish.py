#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fuse_cap_polish_ext.py — FuseCap-PL (REWRITE 1→1) ze strumieniowym zapisem i filtrami jakości:
- Generuje JEDNO płynne zdanie po polsku (utrzymuje sens; dopuszcza 1–3 słowa ekstra, głównie z OCR).
- Nie używa detekcji obiektów do promptu (żeby nie „wpychać” nart/snowboardu, gdy to sanki).
- Twarde filtry: semantyka (BERTScore XLM-R), spójność płci, spójność sprzętu (sanki vs narty/snowboard),
  poprawiona detekcja nazw własnych (nie myli kapitalizacji na początku zdania z nazwą).
- Dynamiczna kontrola długości: cand ≥ max(min_words, base + delta), oraz górny limit.
- Retry do N prób (z innymi beamami/samplowaniem), stream JSONL + per-image + checkpoint.

Wymagane:
pip install ultralytics easyocr transformers accelerate sentence-transformers bert-score sentencepiece protobuf pillow tqdm numpy open_clip_torch
# (opcjonalnie, ale przydatne do mCLIP obraz->tekst) – dla CUDA 12.4:
# pip install --index-url https://download.pytorch.org/whl/cu124 torchvision
"""

import argparse, json, os, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

# OCR (YOLO nie używamy do promptu — można zostawić dla przyszłych rozszerzeń)
import easyocr

# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# CLIP (diagnostycznie) + tekst-tekst
from sentence_transformers import SentenceTransformer, util
import open_clip

try:
    from bert_score import score as bertscore_score
    HAS_BERTSCORE = True
except Exception:
    HAS_BERTSCORE = False


# ------------ utils ------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_image_id(x) -> str:
    s = str(x).strip()
    if not s.lower().endswith((".jpg", ".jpeg", ".png")):
        s += ".jpg"
    return s

def pick_device(name: str) -> str:
    if name == "cuda" and torch.cuda.is_available(): return "cuda"
    if name == "mps" and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def ocr_easy(reader, image_path: str, min_prob=0.55, max_len=32, limit=8) -> List[str]:
    """
    Lightweight OCR wrapper for EasyOCR.
    - Filters out low-confidence (< min_prob), overly long tokens (> max_len), and duplicates (case-insensitive).
    - Returns up to `limit` unique short tokens found in the image.
    """
    try:
        res = reader.readtext(image_path)
    except Exception:
        return []
    out, seen = [], set()
    for _, text, prob in res:
        if not text or prob is None or prob < min_prob:
            continue
        t = str(text).strip()
        if not t or len(t) > max_len:
            continue
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
        if len(out) >= limit:
            break
    return out

# ------------ prompt & decoding ------------
EXPAND_PROMPT_PL = """[INSTRUKCJA]
Przepisz poniższy podpis po polsku jako JEDNO płynne zdanie (≈18–28 słów), poprawne gramatycznie.
Nie zmieniaj płci ani liczby osób; nie zmieniaj rodzaju aktywności i sprzętu (np. sanki pozostają sankami).
Możesz dodać 1–3 słowa wyłącznie, jeśli wynikają z OCR. Nie dodawaj imion, wieku, marek, dat ani lokalizacji.
Zwróć TYLKO to jedno zdanie, bez komentarzy, list, nagłówków i cytatów.

[ORYGINAŁ]
{base}

[OCR (opcjonalnie)]
{ocr}

<OUTPUT>
"""

_ECHO_CUT_KEYS = ("[INSTRUKCJA]", "[ORYGINAŁ]", "[OCR", "Wynik (rozszerzona", "Wynik (same dodatkowe")
_SENT_END = re.compile(r"[.!?]")
_META_PAT  = re.compile(r"(\[[^\]]*\]|<[^>]*>|^[-*•]\s+|^\d+\.\s+|[A-Za-zĄĆĘŁŃÓŚŹŻąśćęłńóśźż]+:\s+)")
_BRACKETS  = re.compile(r"[\[\]<>]")

def _extract_output(decoded: str, prompt: str) -> str:
    txt = decoded
    if prompt in txt: txt = txt.split(prompt, 1)[1]
    if "<OUTPUT>" in txt: txt = txt.split("<OUTPUT>", 1)[1]
    if "</OUTPUT>" in txt: txt = txt.split("</OUTPUT>", 1)[0]
    for key in _ECHO_CUT_KEYS:
        if key in txt:
            txt = txt.split(key, 1)[-1]
    return txt.strip().strip('"').strip()

def one_sentence(text: str) -> str:
    m = _SENT_END.search(text)
    s = text.strip() if not m else text[:m.start()+1].strip()
    if s and not s.endswith("."): s += "."
    return re.sub(r"\s+", " ", s)

def sanitize_candidate(raw: str) -> str:
    t = raw.replace("\n", " ").strip()
    t = _META_PAT.sub("", t).strip()
    t = _BRACKETS.sub("", t)
    t = one_sentence(t)
    t = t.strip(' "\'`')
    return t

def token_set_pl(text: str) -> set:
    return set(re.findall(r"[a-ząćęłńóśźż]+", text.lower()))

def words_ok(base_n: int, cand_n: int, min_words: int, delta_words: int, max_words: int) -> bool:
    return (cand_n >= max(min_words, base_n + delta_words)) and (cand_n <= max_words)


# ------------ consistency guards ------------
# Płeć (prosto i skutecznie)
MALE = {"chłopiec","chlopiec","mężczyzna","mezczyzna","pan","chłopak","chlopak","facet"}
FEMALE = {"dziewczynka","kobieta","pani","dziewczyna"}
def detect_gender(tokens: set) -> str:
    if tokens & MALE: return "male"
    if tokens & FEMALE: return "female"
    return "neutral"

def violates_gender(base_txt: str, cand_txt: str) -> bool:
    b = token_set_pl(base_txt); c = token_set_pl(cand_txt)
    gb, gc = detect_gender(b), detect_gender(c)
    if gb == "neutral" and gc != "neutral": return True
    if gb == "male" and gc == "female": return True
    if gb == "female" and gc == "male": return True
    return False

# Sprzęt: sanki vs narty/snowboard (bez YOLO, tylko z tekstu)
EQUIP = {
    "sled": {"sanki","sanek","saneczki","saneczkach","sania","saniach","sanie","jabłuszko","jabluszko","jabłuszku","jabluszku","ślizg","slizg"},
    "ski": {"narty","narciarz","narciarze","narciarka","narciarski","narciarska","narciarskie"},
    "snowboard": {"snowboard","snowboardzista","snowboardzistka","deska"}
}
def equip_groups(tokens: set) -> set:
    groups = set()
    for g, vocab in EQUIP.items():
        if tokens & vocab: groups.add(g)
    return groups

def violates_equipment(base_txt: str, cand_txt: str) -> bool:
    b = equip_groups(token_set_pl(base_txt)); c = equip_groups(token_set_pl(cand_txt))
    if not b:  # w bazie nie ma sprzętu → nie pozwalaj na wprowadzanie innego niż neutralny „dziecko/siedzi etc.”
        return bool(c)  # w ogóle nie dodawaj sprzętu
    return len(c - b) > 0  # kandydat nie może wprowadzać innej grupy niż w bazie

# „Nazwy własne” – poprawka: ignoruj pierwszy wyraz w zdaniu + zezwól, jeśli (lower) jest w bazie lub OCR
_CAPWORD = re.compile(r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\b")
def has_new_proper_nouns(base: str, cand: str, allow_ocr: List[str]) -> bool:
    base_l = token_set_pl(base)
    allow_l = token_set_pl(" ".join(allow_ocr))
    # rozbij na zdania, w każdym pomiń pierwszy token
    sentences = re.split(r"(?<=[.!?])\s+", cand.strip())
    cap_words: List[str] = []
    for s in sentences:
        toks = s.strip().split()
        if not toks: continue
        rest = " ".join(toks[1:])  # pomijamy pierwszy wyraz zdania
        cap_words += _CAPWORD.findall(rest)
    for w in cap_words:
        wl = w.lower()
        if wl not in base_l and wl not in allow_l:
            return True
    return False


# ------------ scoring ------------
def clip_score_multilingual(st_clip, image_path: str, caption_pl: str,
                            oclip_model=None, oclip_tokenizer=None, oclip_preprocess=None, device="cuda") -> float:
    try:
        img = Image.open(image_path).convert("RGB")
        e_img = st_clip.encode([img], convert_to_tensor=True, normalize_embeddings=True)
        e_txt = st_clip.encode([caption_pl], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(util.cos_sim(e_img, e_txt)[0,0])
        if sim > 0.0 and np.isfinite(sim):
            return sim
    except Exception:
        pass
    try:
        img = Image.open(image_path).convert("RGB")
        image = oclip_preprocess(img).unsqueeze(0).to(device)
        text  = oclip_tokenizer([caption_pl]).to(device)
        with torch.no_grad():
            i_feat = oclip_model.encode_image(image)
            t_feat = oclip_model.encode_text(text)
            i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            sim = (i_feat @ t_feat.T).float().item()
        return float(sim)
    except Exception:
        return 0.0

def text_text_similarity(base: str, hyp: str, st_text: Optional[SentenceTransformer]) -> float:
    if HAS_BERTSCORE:
        P, R, F1 = bertscore_score([hyp], [base], model_type="xlm-roberta-large", rescale_with_baseline=False)
        return float(F1[0])
    else:
        v = st_text.encode([base, hyp], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(v[0:1], v[1:2])[0,0])


# ------------ generation ------------
def fuse_one(model, tok, base: str, ocr_tokens: List[str],
             max_new_tokens=48, temperature=0.6, top_p=0.9,
             num_beams=3, do_sample=False, repetition_penalty=1.1,
             device="cuda") -> str:
    ocr_txt = ", ".join(ocr_tokens) if ocr_tokens else "brak"
    prompt = EXPAND_PROMPT_PL.format(base=base, ocr=ocr_txt)
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        eos_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
        repetition_penalty=repetition_penalty,
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=max(1, num_beams), no_repeat_ngram_size=3))
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    raw = tok.decode(out[0], skip_special_tokens=True)
    return sanitize_candidate(_extract_output(raw, prompt))


# ------------ main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ocr_langs", default="pl")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu","mps"])

    # LLM
    ap.add_argument("--llm_model_id", default="speakleash/Bielik-7B-Instruct-v0.1")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--retries", type=int, default=2)

    # Progi filtrów
    ap.add_argument("--textsim_threshold", type=float, default=0.90)
    ap.add_argument("--clip_threshold", type=float, default=0.18)
    ap.add_argument("--clip_as_gate", action="store_true")  # normalnie CLIP nie blokuje
    ap.add_argument("--expand_min_delta_words", type=int, default=1)
    ap.add_argument("--min_words", type=int, default=12)
    ap.add_argument("--max_words", type=int, default=40)
    ap.add_argument("--keep_original", action="store_true")

    # stream/checkpoint/debug
    ap.add_argument("--jsonl_out", default=None)
    ap.add_argument("--stream_dir", default=None)
    ap.add_argument("--checkpoint_every", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--debug_every", type=int, default=100)
    ap.add_argument("--debug_dump_first_k", type=int, default=0)

    args = ap.parse_args()
    args.device = pick_device(args.device)

    images_root = Path(args.images_root)
    data = load_json(args.input_json)
    if args.limit: data = data[:args.limit]

    # MODELE (OCR + CLIP diagnostycznie)
    reader = easyocr.Reader([l.strip() for l in args.ocr_langs.split(",") if l.strip()],
                            gpu=(args.device=="cuda"))
    st_clip = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device=args.device)
    oclip_model, _, oclip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=args.device)
    oclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    st_text = None if HAS_BERTSCORE else SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=args.device)

    tok = AutoTokenizer.from_pretrained(args.llm_model_id, use_fast=True, trust_remote_code=True)
    if args.device == "cuda" and not args.no_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.float16 if args.device in ("cuda","mps") else torch.float32
        device_map = {"": args.device} if args.device in ("cuda","mps") else None
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    out = []
    processed = 0
    reject_stats: Dict[str,int] = {"len":0,"textsim":0,"clip":0,"proper":0,"digits":0,"gender":0,"equip":0,"noise":0,"echo":0,"retry":0,"ok":0}

    if args.stream_dir: Path(args.stream_dir).mkdir(parents=True, exist_ok=True)
    if args.jsonl_out: Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(data, desc="FuseCap-PL (rewrite 1→1)"):
        raw_id = rec.get("image_id")
        img_file = normalize_image_id(raw_id)
        caps = [c.strip() for c in (rec.get("captions", []) or []) if isinstance(c, str) and c.strip()]
        img_path = images_root / img_file
        item: Dict = {"image_id": img_file}

        if not img_path.exists() or not caps:
            if args.keep_original: item["captions_original"] = caps
            item["captions"] = caps
            out.append(item)
            if args.jsonl_out:
                with open(args.jsonl_out, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(item, ensure_ascii=False) + "\n"); jf.flush(); os.fsync(jf.fileno())
            if args.stream_dir:
                with open(Path(args.stream_dir)/f"{Path(img_file).stem}.json", "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
            processed += 1
            if args.checkpoint_every and args.checkpoint and processed % args.checkpoint_every == 0:
                with open(args.checkpoint, "w", encoding="utf-8") as cf:
                    json.dump(out, cf, ensure_ascii=False, indent=2)
            if args.debug_every and processed % args.debug_every == 0:
                print(f"[debug] rejects so far: {reject_stats}", flush=True)
            continue

        # OCR tokens (opcjonalnie do lekkiego wzbogacenia)
        ocr_tokens = ocr_easy(reader, str(img_path))

        new_caps = []
        debug_dump = []

        for base in caps:
            attempts = 0
            reason = None
            cand = ""
            sim_text = None; sim_clip = None

            while attempts <= args.retries:
                attempts += 1
                cand = fuse_one(
                    model, tok, base, ocr_tokens,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    num_beams=args.num_beams, do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                    device=args.device
                )

                if not cand:
                    reason = "echo"; continue
                if any(k in cand for k in ("INSTRUKCJA","ORYGINAŁ","<OUTPUT")):
                    reason = "echo"; continue
                if any(ch in cand for ch in "<>[]{}|*`~"):
                    reason = "noise"; continue

                # spójność sprzętu i płci przed semantyką
                if violates_equipment(base, cand):
                    reason = "equip"; continue
                if violates_gender(base, cand):
                    reason = "gender"; continue

                # nazwy własne — poprawiona heurystyka
                if has_new_proper_nouns(base, cand, ocr_tokens):
                    reason = "proper"; continue

                # długość
                if not words_ok(len(base.split()), len(cand.split()), args.min_words, args.expand_min_delta_words, args.max_words):
                    reason = "len"; continue

                # semantyka (główna bramka)
                sim_text = text_text_similarity(base, cand, st_text)
                if sim_text < args.textsim_threshold:
                    reason = "textsim"; continue

                # CLIP tylko jeśli zażądasz bramki
                try:
                    sim_clip = clip_score_multilingual(
                        SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device=args.device),
                        str(img_path), cand,
                        oclip_model=oclip_model,
                        oclip_tokenizer=oclip_tokenizer,
                        oclip_preprocess=oclip_preprocess,
                        device=args.device
                    )
                except Exception:
                    sim_clip = 0.0
                if args.clip_as_gate and sim_clip < args.clip_threshold:
                    reason = "clip"; continue

                reason = None
                break

            if reason:
                reject_stats[reason] = reject_stats.get(reason, 0) + 1
                new_caps.append(base)
                if attempts > 1: reject_stats["retry"] += 1
            else:
                reject_stats["ok"] += 1
                new_caps.append(cand)

            if args.debug_dump_first_k and processed < args.debug_dump_first_k:
                debug_dump.append({
                    "base": base, "cand": cand, "reason": reason, "attempts": attempts,
                    "words_base": len(base.split()), "words_cand": len(cand.split()),
                    "sim_text": sim_text, "sim_clip": sim_clip,
                    "ocr": ocr_tokens[:4],
                })

        if args.keep_original: item["captions_original"] = caps
        item["captions"] = new_caps
        if debug_dump: item["_debug"] = debug_dump
        out.append(item)

        if args.jsonl_out:
            with open(args.jsonl_out, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(item, ensure_ascii=False) + "\n"); jf.flush(); os.fsync(jf.fileno())
        if args.stream_dir:
            with open(Path(args.stream_dir)/f"{Path(img_file).stem}.json", "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

        processed += 1
        if args.checkpoint_every and args.checkpoint and processed % args.checkpoint_every == 0:
            with open(args.checkpoint, "w", encoding="utf-8") as cf:
                json.dump(out, cf, ensure_ascii=False, indent=2)
        if args.debug_every and processed % args.debug_every == 0:
            print(f"[debug] [{processed}/{len(data)}] rejects so far: {reject_stats}", flush=True)

    save_json(out, args.output_json)
    print(f"[OK] zapisano {len(out)} rekordów -> {args.output_json}")
    print(f"[summary] rejects: {reject_stats}; BERTScore used: {HAS_BERTSCORE}; device: {args.device}")

if __name__ == "__main__":
    main()
