import sys, os, json
from collections import defaultdict, Counter

def main(path):
    n_lines = 0
    n_ok = 0
    n_missing_img = 0
    bad_structure = 0

    seen_pairs = Counter()
    per_img_caps = defaultdict(list)
    img_set = set()

    def norm(s): 
        return " ".join((s or "").strip().split())

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception as e:
                print(f"[ERROR] Linia {n_lines}: błąd JSON: {e}")
                continue

            img = ex.get("image", "")
            conv = ex.get("conversations", [])
            if not img or not isinstance(conv, list) or len(conv) < 2:
                bad_structure += 1
                continue
            if not os.path.exists(img):
                n_missing_img += 1

            ok = (conv[0].get("from") == "human" and conv[1].get("from") == "gpt")
            if not ok:
                bad_structure += 1
                continue

            cap = norm(conv[1].get("value", ""))
            if not cap:
                bad_structure += 1
                continue

            n_ok += 1
            img_set.add(img)
            per_img_caps[img].append(cap)
            seen_pairs[(img, cap)] += 1

    total_imgs = len(img_set)
    total_examples = n_ok
    total_dupe_pairs = sum(c for (k,c) in seen_pairs.items() if c > 1)
    caps_len_chars = [len(c) for caps in per_img_caps.values() for c in caps]
    caps_len_words = [len(c.split()) for caps in per_img_caps.values() for c in caps]

    print(f"[INFO] Plik: {path}")
    print(f"[INFO] Przykłady (linie ok): {total_examples} / {n_lines}")
    print(f"[INFO] Unikalnych obrazów: {total_imgs}")
    print(f"[INFO] Brakujący plik obrazu: {n_missing_img}")
    print(f"[INFO] Błędna struktura (np. conversations): {bad_structure}")
    if caps_len_chars:
        avg_chars = sum(caps_len_chars)/len(caps_len_chars)
        p95_chars = sorted(caps_len_chars)[int(0.95*len(caps_len_chars))-1]
        avg_words = sum(caps_len_words)/len(caps_len_words)
        print(f"[INFO] Śr. długość podpisu: {avg_chars:.1f} znaków, {avg_words:.1f} słów; p95 znaków: {p95_chars}")
    print(f"[INFO] Duplikaty (identyczny obraz+podpis): {total_dupe_pairs}")

    dupes = [(k,v) for k,v in seen_pairs.items() if v>1]
    dupes.sort(key=lambda kv: kv[1], reverse=True)
    if dupes[:5]:
        print("\n[INFO] Top duplikaty (obraz, podpis) x count:")
        for (img, cap), cnt in dupes[:5]:
            print(f"  x{cnt}  {os.path.basename(img)}  |  {cap[:120]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python validate_jsonl.py data/train.jsonl")
        sys.exit(1)
    main(sys.argv[1])
