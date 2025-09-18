import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

def load_id_list(path):
    keep = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if not name.lower().endswith(".jpg"):
                name += ".jpg"
            keep.add(name)
    return keep

def main():
    parser = argparse.ArgumentParser(description="Build JSON for mPLUG from recastai/flickr30k-augmented-caption")
    parser.add_argument("--out", required=True, help="Ścieżka wyjściowa do pliku JSON (np. flickr30k_augmented_train.json)")
    parser.add_argument("--split_ids", default=None, help="Opcjonalnie: ścieżka do pliku txt z listą filename'ów (np. train_ids.txt)")
    parser.add_argument("--max_captions_per_image", type=int, default=None,
                        help="Opcjonalnie: maksymalna liczba podpisów na obraz (np. 5). Jeśli None — wszystkie po deduplikacji.")
    parser.add_argument("--min_len", type=int, default=3, help="Minimalna liczba znaków podpisu po strip() (dla sanity).")
    parser.add_argument("--dry_run", action="store_true", help="Nie zapisuje pliku, tylko wypisuje statystyki.")
    args = parser.parse_args()

    ds = load_dataset("recastai/flickr30k-augmented-caption", split="train")

    keep_ids = None
    if args.split_ids:
        if not os.path.exists(args.split_ids):
            raise FileNotFoundError(f"Nie znaleziono pliku splitu: {args.split_ids}")
        keep_ids = load_id_list(args.split_ids)
        print(f"[INFO] Wczytano {len(keep_ids)} ID z {args.split_ids}.")

    grouped = defaultdict(list)
    for ex in tqdm(ds, desc="Grupowanie"):
        fname = ex.get("filename")
        cap = ex.get("caption", "")
        if not fname or not isinstance(cap, str):
            continue
        cap_clean = cap.strip()
        if len(cap_clean) < args.min_len:
            continue
        if keep_ids is not None and fname not in keep_ids:
            continue
        if cap_clean not in grouped[fname]:
            grouped[fname].append(cap_clean)

    if args.max_captions_per_image is not None and args.max_captions_per_image > 0:
        for k in grouped:
            grouped[k] = grouped[k][:args.max_captions_per_image]

    records = []
    for fname in sorted(grouped.keys()):
        caps = grouped[fname]
        if not caps:
            continue
        records.append({
            "image_id": fname,
            "captions": caps
        })

    num_images = len(records)
    num_caps_total = sum(len(r["captions"]) for r in records)
    caps_per_image_avg = round(num_caps_total / num_images, 3) if num_images else 0.0

    print(f"[INFO] Obrazy w wyjściu: {num_images}")
    print(f"[INFO] Podpisy łącznie: {num_caps_total}")
    print(f"[INFO] Średnio podpisów/obraz: {caps_per_image_avg}")

    if not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[OK] Zapisano JSON do: {args.out}")

if __name__ == "__main__":
    main()
