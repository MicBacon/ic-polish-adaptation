import json
from pathlib import Path
import pandas as pd
import numpy as np
from MetricComputer import MetricComputer

QWEN_JSON = "translate_evaluate_results/bing/Qwen2_5-VL_raw_en_test_hq.json"
MPLUG_JSON = "translate_evaluate_results/bing/mPLUG_flickr_en_test_hq.json"

IMAGE_ROOT = "/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/shared/data/flickr30k/Images"

BLEU1_SUS_THRESHOLD = 0.1
CLIP_SUS_THRESHOLD  = 17.0

OUT_CSV = "sus_cases_b1_clip_hq.csv"

def load_json_array(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data

def resolve_image_path(p):
    p = p or ""
    pth = Path(p)
    if pth.exists():
        return str(pth)
    if IMAGE_ROOT:
        return str(Path(IMAGE_ROOT) / p)
    return str(pth)

def to_df(model_name, arr):
    rows = []
    for it in arr:
        rows.append({
            "model": model_name,
            "img_id": it.get("img_id"),
            "img_path": resolve_image_path(it.get("img_path","")),
            "pred_pl": it.get("pl_prediction","") or "",
            "refs_pl": it.get("references_pl", []) or [],
        })
    return pd.DataFrame(rows)

def compute_scores_with_mc(df, mc):
    bleu_scores = []

    for _, row in df.iterrows():
        #if row['model'] == 'mPLUG':
        #    bleu_scores.append(float(0.1))
        #else:
        bleu = mc.bleu1_single(row["pred_pl"], row["refs_pl"])
        bleu_scores.append(float(bleu) if bleu is not None else 0.0)
    df["BLEU1"] = bleu_scores

    clip_scores = []
    for _, row in df.iterrows():
        cs = mc.clipscore_single(row["img_path"], row["pred_pl"])
        clip_scores.append(float(cs) if cs is not None else np.nan)
    df["CLIPScore"] = clip_scores

    df["sus_bleu"] = df["BLEU1"] < BLEU1_SUS_THRESHOLD
    if df["CLIPScore"].notna().any():
        df["sus_clip"] = df["CLIPScore"] < CLIP_SUS_THRESHOLD
    else:
        df["sus_clip"] = False
    df["sus_any"] = df["sus_bleu"] | df["sus_clip"]
    return df

def side_by_side(qwen, mplug):
    qa = qwen.rename(columns={
        "pred_pl":"pred_pl_qwen", "BLEU1":"BLEU1_qwen", "CLIPScore":"CLIP_qwen",
        "sus_bleu":"sus_bleu_qwen", "sus_clip":"sus_clip_qwen", "sus_any":"sus_any_qwen"
    })
    ma = mplug.rename(columns={
        "pred_pl":"pred_pl_mplug", "BLEU1":"BLEU1_mplug", "CLIPScore":"CLIP_mplug",
        "sus_bleu":"sus_bleu_mplug", "sus_clip":"sus_clip_mplug", "sus_any":"sus_any_mplug"
    })
    merged = qa.merge(ma, on=["img_id"], how="outer", suffixes=("_q","_m"))

    if "img_path_x" in merged.columns and "img_path_y" in merged.columns:
        merged["img_path"] = merged["img_path_x"].fillna(merged["img_path_y"])
    elif "img_path_x" in merged.columns:
        merged["img_path"] = merged["img_path_x"]
    elif "img_path_y" in merged.columns:
        merged["img_path"] = merged["img_path_y"]
    else:
        merged["img_path"] = ""

    if "refs_pl_x" in merged.columns and "refs_pl_y" in merged.columns:
        merged["refs_pl"]  = merged["refs_pl_x"].where(merged["refs_pl_x"].notna(), merged["refs_pl_y"])
    elif "refs_pl_x" in merged.columns:
        merged["refs_pl"]  = merged["refs_pl_x"]
    elif "refs_pl_y" in merged.columns:
        merged["refs_pl"]  = merged["refs_pl_y"]
    else:
        merged["refs_pl"]  = [[] for _ in range(len(merged))]

    drop_cols = [c for c in merged.columns if (c.endswith("_x") or c.endswith("_y")) and c not in ("img_path","refs_pl")]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged

def main():
    qwen_arr  = load_json_array(QWEN_JSON)
    mplug_arr = load_json_array(MPLUG_JSON)

    # last object = metric values
    qwen_df  = to_df("Qwen2.5-VL", qwen_arr)[:-1]
    mplug_df = to_df("mPLUG",      mplug_arr)[:-1]

    mc = MetricComputer()

    qwen_df  = compute_scores_with_mc(qwen_df, mc)
    mplug_df = compute_scores_with_mc(mplug_df, mc)

    merged = side_by_side(qwen_df, mplug_df)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    
    print("Zapisano wynik do", OUT_CSV)

if __name__ == "__main__":
    main()
