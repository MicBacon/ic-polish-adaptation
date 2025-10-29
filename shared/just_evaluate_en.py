# en_en_eval.py
import os
import json
import jsonlines
from typing import List, Tuple, Dict, Any
from MetricComputer import MetricComputer

INPUT_EN_PRED = [#'../mPLUG/output/flickr30k/eval_after_flickr_finetune_test_hq.json',
                  #'../mPLUG/output/flickr30k/eval_after_flickr_finetune_test_std.json',
                  '../mPLUG/output/flickr30k/eval_after_full_finetune_test_hq.json',
                  '../mPLUG/output/flickr30k/eval_after_full_finetune_test_std.json',
                  ]
EN_REF_PATH   = ['data/flickr30k/flickr30kEnglish_captions_test_hq.json', 
                 'data/flickr30k/flickr30kEnglish_captions_test_std.json',
                 ]
OUT_METRICS   = ['just_evaluate_en_results/mPLUG_FULL_metrics_EN_EN_hq.json', 
                 'just_evaluate_en_results/mPLUG_FULL_metrics_EN_EN_std.json',
                 ]

IMAGE_ROOT = "/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/shared/data/flickr30k/Images"
EXTS = ("jpg", "jpeg", "png")

def norm_img_id(x: str) -> str:
    x = str(x).strip()
    x = os.path.basename(x)
    return x.split('.')[0] if '.' in x else x

def id_to_path(stem: str) -> str:
    for ext in EXTS:
        p = os.path.join(IMAGE_ROOT, f"{stem}.{ext}")
        if os.path.isfile(p):
            return p
    return ""

def load_en_refs(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        items = json.load(f)
    rows = []
    for it in items:
        kid  = norm_img_id(it.get('image_id', it.get('id', '')))
        caps = it.get('captions') or it.get('references') or it.get('refs') or []
        caps = [str(c).strip().lower().removesuffix('.') for c in caps if isinstance(c, str)]
        if kid and caps:
            rows.append({"image_id": kid, "captions": caps})
    return rows

def load_en_preds(path: str) -> List[Dict[str, Any]]:
    rows = []
    if path.endswith('.jsonl'):
        with jsonlines.open(path) as r:
            for row in r:
                kid = norm_img_id(row.get('image_id', row.get('id', row.get('image_path', ''))))
                pred = row.get('prediction', '')
                if kid and isinstance(pred, str):
                    rows.append({"image_id": kid, "prediction": pred.strip(), "image_path": id_to_path(kid)})
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        for row in data:
            kid = norm_img_id(row.get('image_id', row.get('question_id', row.get('id', ''))))
            pred = row.get('pred_caption', row.get('prediction', ''))
            if kid and isinstance(pred, str):
                rows.append({"image_id": kid, "prediction": pred.strip(), "image_path": id_to_path(kid)})
    return rows

def main():
    for input_en_pred, en_ref_path, out_metrics in zip(INPUT_EN_PRED, EN_REF_PATH, OUT_METRICS):
        ref_rows  = load_en_refs(en_ref_path)   
        pred_rows = load_en_preds(input_en_pred)

        print("refs:", len(ref_rows), "preds:", len(pred_rows))

        mc = MetricComputer(
            bert_lang=None,
            clip_model_name="ViT-B-32",
            clip_pretrained="laion2b_s34b_b79k",
            clip_prompt="",
            bert_model_type="microsoft/deberta-xlarge-mnli"
        )

        metrics = mc.compute_metrics_by_id(
            pred_rows=pred_rows,
            ref_rows=ref_rows,
            image_root=IMAGE_ROOT,
            id_key_pred="image_id",   
            id_key_ref="image_id",
            cap_key_pred="prediction",
            caps_key_ref="captions",
            fast=False
        )

        with open(out_metrics, 'w') as f:
            json.dump({"overall_metric_score": metrics}, f, ensure_ascii=False, indent=2)
        print(metrics)

if __name__ == "__main__":
    main()