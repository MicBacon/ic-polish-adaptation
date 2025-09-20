import argparse
import json
import os
import sys
import unicodedata
import random

def normalize_text(s: str, to_lower: bool=False) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    if to_lower:
        s = s.lower()
    s = " ".join(s.split())
    return s

def find_image_path(images_dir: str, image_id: str, ext_priority) -> str:
    for ext in ext_priority:
        cand = os.path.join(images_dir, f"{image_id}{ext}")
        if os.path.exists(cand):
            return cand
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Ścieżka do wejściowego pliku JSON (lista obiektów z image_id i captions).")
    ap.add_argument("--images_dir", required=True, help="Folder z obrazami.")
    ap.add_argument("--out_jsonl", required=True, help="Ścieżka wyjściowa JSONL.")
    ap.add_argument("--prompt", default="<image>\nOpisz obraz po polsku jednym zwięzłym zdaniem.", help="Prompt użytkownika (domyślnie PL).")
    ap.add_argument("--ext_priority", nargs="+", default=[".jpg", ".png", ".jpeg"], help="Priorytet rozszerzeń do wyszukiwania plików.")
    ap.add_argument("--strip_empty", action="store_true", help="Pomiń puste podpisy.")
    ap.add_argument("--lowercase", action="store_true", help="Zamień podpisy na małe litery.")
    ap.add_argument("--max_caps_per_image", type=int, default=0, help="Maks. liczba podpisów na obraz (0 = bez limitu).")
    ap.add_argument("--shuffle", action="store_true", help="Tasuj kolejność przykładów (deterministycznie).")
    ap.add_argument("--seed", type=int, default=42, help="Ziarno RNG dla --shuffle.")
    args = ap.parse_args()

    try:
        with open(args.in_json, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        print(f"[ERROR] Nie udało się wczytać {args.in_json}: {e}", file=sys.stderr)
        sys.exit(1)

    total_items = 0
    total_captions = 0
    total_written = 0
    total_missing_images = 0
    total_empty_skipped = 0

    examples = []

    for rec in records:
        image_id = str(rec.get("image_id", "")).strip()
        caps = rec.get("captions", []) or []
        total_items += 1
        total_captions += len(caps)

        img_path = find_image_path(args.images_dir, image_id, args.ext_priority)
        if not img_path:
            total_missing_images += 1
            print(f"[WARN] Brak pliku obrazu dla image_id={image_id}", file=sys.stderr)
            continue

        if args.max_caps_per_image and len(caps) > args.max_caps_per_image:
            caps = caps[:args.max_caps_per_image]

        for cap in caps:
            if not isinstance(cap, str):
                continue
            cap_norm = normalize_text(cap, to_lower=args.lowercase)
            if args.strip_empty and not cap_norm:
                total_empty_skipped += 1
                continue

            ex = {
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": f"{args.prompt}"},
                    {"from": "gpt",   "value": cap_norm}
                ]
            }
            examples.append(ex)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(examples)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for ex in examples:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            total_written += 1

    print(f"[INFO] Wejście: {args.in_json}")
    print(f"[INFO] Znalazłem rekordów: {total_items}, podpisów łącznie: {total_captions}")
    print(f"[INFO] Zapisane próbki: {total_written}")
    if total_missing_images:
        print(f"[WARN] Próbki bez plików obrazu: {total_missing_images} (sprawdź rozszerzenia/ścieżki)")
    if args.strip_empty and total_empty_skipped:
        print(f"[INFO] Pominięte puste podpisy: {total_empty_skipped}")
    print(f"[DONE] Wyjście: {args.out_jsonl}")

if __name__ == "__main__":
    main()
