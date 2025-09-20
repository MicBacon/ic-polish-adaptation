import sys, json, random, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Ścieżka do pliku .jsonl")
    ap.add_argument("--n", type=int, default=5, help="Liczba przykładów do podglądu")
    ap.add_argument("--seed", type=int, default=42, help="Ziarno RNG")
    args = ap.parse_args()

    lines = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex = json.loads(ln)
            except:
                continue
            lines.append(ex)

    random.seed(args.seed)
    random.shuffle(lines)
    for i, ex in enumerate(lines[:args.n], 1):
        image = ex.get("image")
        conv = ex.get("conversations", [])
        human = conv[0]["value"] if conv and len(conv)>0 else ""
        gpt = conv[1]["value"] if conv and len(conv)>1 else ""
        print(f"\n=== Przykład {i} ===")
        print(f"Obraz: {image}  ({'OK' if image and os.path.exists(image) else 'BRAK'})")
        print(f"Human: {human}")
        print(f"GPT  : {gpt}")

if __name__ == "__main__":
    main()
