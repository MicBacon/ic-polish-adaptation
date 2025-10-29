import json, jsonlines
import translators as ts
import os, re
import time
import random
import stanza
from typing import Any, Dict, List, Tuple, Optional
from bert_score import BERTScorer
from MetricComputer import MetricComputer

INPUT_PATHS = [
    #'../Qwen2_5-VL/eval_results/raw_en_test_hq/predictions.jsonl',
    #'../Qwen2_5-VL/eval_results/raw_en_test_std/predictions.jsonl',
    #'../mPLUG/output/flickr30k/eval_after_flickr_finetune_test_hq.json',
    #'../mPLUG/output/flickr30k/eval_after_flickr_finetune_test_std.json',
    '../mPLUG/output/flickr30k/eval_after_full_finetune_test_hq.json',
    #'../mPLUG/output/flickr30k/eval_after_full_finetune_test_std.json',
]
DST_FILES = [
    #'Qwen2_5-VL_raw_en_test_hq.json',
    #'Qwen2_5-VL_raw_en_test_std.json',
    #'mPLUG_flickr_en_test_hq.json',
    #'mPLUG_flickr_en_test_std.json',
    'mPLUG_FULL_test_hq.json',
    #'mPLUG_FULL_test_std.json',
]
RESULTS_FOLDER = 'translate_evaluate_results/'

PL_REFERENCES_PATH = 'data/flickr30k/flickr30kPolish_captions_test_hq.json'

IMAGE_ROOT = '/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/shared/data/flickr30k/Images'

TRANSLATOR = 'bing'
PREACCELERATE = True
RATE_LIMIT_SLEEP = 0.6 if TRANSLATOR == 'bing' else 0.0

nlp = stanza.Pipeline(
    lang="pl",
    processors="tokenize,lemma",
    tokenize_no_ssplit=True,
    use_gpu=False
)

_ws = re.compile(r"\s+")
_id_digits = re.compile(r"(\d{6,})")

def norm_space(s: str) -> str:
    return _ws.sub(" ", s).strip()

def lemmatize_pl(text: str) -> str:
    if not text:
        return ""
    doc = nlp(text.lower())
    lemmas = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.lemma:
                lemmas.append(w.lemma.strip())
    return norm_space(" ".join(lemmas))

def normalize_pl(s: str) -> str:
    s = ' '.join(s.strip().split())
    if s and s[-1] not in '.!?':
        s += '.'
    return s

def translate_with_retry(text: str,
                         translator: str,
                         src: str = 'en',
                         dst: str = 'pl',
                         max_retries: int = 6,
                         base_delay: float = 1.0,
                         backoff: float = 1.7,
                         jitter: float = 0.3) -> str:
    last_exc = None
    for attempt in range(max_retries):
        try:
            out = ts.translate_text(
                text,
                translator=translator,
                from_language=src,
                to_language=dst,
                if_use_preacceleration=PREACCELERATE
            )
            return normalize_pl(out)
        except Exception as e:
            last_exc = e
            sleep_s = base_delay * (backoff ** attempt) + random.uniform(0, jitter)
            print(f"[WARN] Translation failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"failed after {max_retries} retries ({translator}): {last_exc}")

def extract_prediction(item: Dict[str, Any]) -> Optional[str]:
    for k in ['prediction', 'pred_caption', 'caption', 'generated_caption', 'answer']:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k].strip()
    return None

def extract_image_id_str_from_any(item: Dict[str, Any]) -> Optional[str]:
    for k in ['image_id', 'img_id', 'id', 'question_id']:
        if k in item:
            v = item[k]
            if isinstance(v, int):
                return str(v)
            if isinstance(v, str):
                if v.isdigit():
                    return v
                m = _id_digits.search(v)
                if m:
                    return m.group(1)

    if 'question_id' in item and isinstance(item['question_id'], str):
        m = _id_digits.search(item['question_id'])
        if m:
            return m.group(1)

    return None

def build_img_path(image_id_str: str) -> str:
    if image_id_str.endswith('.jpg'):
        return os.path.join(IMAGE_ROOT, f"{image_id_str}")
    else:
        return os.path.join(IMAGE_ROOT, f"{image_id_str}.jpg")

def load_pl_references_map(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        out = {}
        for row in data:
            if 'image_id' in row and 'captions' in row:
                out[str(row['image_id'])] = [normalize_pl(c) for c in row['captions'] if isinstance(c, str)]
        return out
    except Exception as e:
        print(f"[WARN] Could not load PL references from {path}: {e}")
        return {}

def main():
    if PREACCELERATE:
        _ = ts.preaccelerate_and_speedtest()

    bs = BERTScorer(model_type="xlm-roberta-large", rescale_with_baseline=False)

    mc = MetricComputer()

    pl_ref_map = load_pl_references_map(PL_REFERENCES_PATH)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    out_dir = os.path.join(RESULTS_FOLDER, TRANSLATOR)
    os.makedirs(out_dir, exist_ok=True)

    for input_path, dst_file in zip(INPUT_PATHS, DST_FILES):
        print(f"[INFO] Processing: {input_path}")

        all_predictions_pl: List[str] = []
        all_predictions_pl_lem: List[str] = []
        all_references_pl: List[List[str]] = []
        all_references_pl_lem: List[List[str]] = []
        all_img_paths: List[str] = []

        out_fp = open(os.path.join(out_dir, dst_file), "w", encoding="utf-8")
        out_fp.write('[\n')
        wrote_any = False

        def write_record(obj: Dict[str, Any]):
            nonlocal wrote_any
            if wrote_any:
                out_fp.write(',\n')
            json.dump(obj, out_fp, ensure_ascii=False)
            wrote_any = True

        try:
            if input_path.endswith('.jsonl'):
                data_iter = jsonlines.open(input_path)
                for idx, line in enumerate(data_iter, 1):
                    pred_en = line.get('prediction', '') or ''
                    if not pred_en.strip():
                        pred_en = 'a photo'
                    references_pl = line.get('references', []) or []

                    image_id_str = str(line.get('id')) if 'id' in line else extract_image_id_str_from_any(line) or f"unknown_{idx}"
                    img_path = build_img_path(image_id_str)

                    pred_pl = translate_with_retry(pred_en, translator=TRANSLATOR, src='en', dst='pl')
                    P, R, F1 = bs.score([pred_en], [pred_pl])
                    P, R, F1 = float(P[0]), float(R[0]), float(F1[0])

                    pred_pl_lem = lemmatize_pl(pred_pl)
                    refs_pl_lem = [lemmatize_pl(r) for r in references_pl]

                    write_record({
                        "img_id": image_id_str,
                        "img_path": img_path,
                        "en_prediction": pred_en,
                        "pl_prediction": pred_pl,
                        "translation_scores": {"P": P, "R": R, "F1": F1},
                        "references_pl": references_pl
                    })

                    all_predictions_pl.append(pred_pl)
                    all_predictions_pl_lem.append(pred_pl_lem)
                    all_references_pl.append(references_pl)
                    all_references_pl_lem.append(refs_pl_lem)
                    all_img_paths.append(img_path)

                    if RATE_LIMIT_SLEEP > 0:
                        time.sleep(RATE_LIMIT_SLEEP)
                    if idx % 50 == 0:
                        out_fp.flush()

            elif input_path.endswith('.json'):
                with open(input_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                        data = data['data']

                if not isinstance(data, list):
                    raise ValueError("Unsupported JSON structure: expected a list of items")

                for idx, item in enumerate(data, 1):
                    pred_en = extract_prediction(item) or 'a photo'
                    image_id_str = extract_image_id_str_from_any(item) or f"unknown_{idx}"
                    img_path = build_img_path(image_id_str)

                    pred_pl = translate_with_retry(pred_en, translator=TRANSLATOR, src='en', dst='pl')
                    P, R, F1 = bs.score([pred_en], [pred_pl])
                    P, R, F1 = float(P[0]), float(R[0]), float(F1[0])

                    pred_pl_lem = lemmatize_pl(pred_pl)

                    references_pl = pl_ref_map.get(image_id_str, [])
                    refs_pl_lem = [lemmatize_pl(r) for r in references_pl]

                    write_record({
                        "img_id": image_id_str,
                        "img_path": img_path,
                        "en_prediction": pred_en,
                        "pl_prediction": pred_pl,
                        "translation_scores": {"P": P, "R": R, "F1": F1},
                        "references_pl": references_pl
                    })

                    all_predictions_pl.append(pred_pl)
                    all_predictions_pl_lem.append(pred_pl_lem)
                    all_references_pl.append(references_pl)
                    all_references_pl_lem.append(refs_pl_lem)
                    all_img_paths.append(img_path)

                    if RATE_LIMIT_SLEEP > 0:
                        time.sleep(RATE_LIMIT_SLEEP)
                    if idx % 50 == 0:
                        out_fp.flush()

            else:
                raise ValueError(f"Unsupported input file extension for: {input_path}")

            print("[INFO] Translation finished. Computing caption metrics on PL (lemmatized)...")
            metrics = mc.compute_metrics(
                all_predictions_pl, all_references_pl,
                all_predictions_pl_lem,
                all_references_pl_lem,
                all_img_paths
            )
            print(metrics)

            out_fp.write(',\n')
            json.dump({"overall_metric_score": metrics}, out_fp, ensure_ascii=False)
            out_fp.write('\n]')

        except Exception as e:
            print(f"[ERROR] Failed on {input_path}: {e}")
            out_fp.write('\n]')
        finally:
            out_fp.close()

    print("[DONE] All inputs processed.")

if __name__ == "__main__":
    main()