from typing import List, Sequence, Optional, Dict, Any, Union, Tuple
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import math, os

tokenizer = PTBTokenizer()

def _norm_img_id(x: str) -> str:
    x = str(x).strip()
    x = os.path.basename(x)
    return x.split('.')[0] if '.' in x else x

def _id_to_path(stem: str, image_root: str, exts=("jpg","jpeg","png")) -> str:
    for ext in exts:
        p = os.path.join(image_root, f"{stem}.{ext}")
        if os.path.isfile(p):
            return p
    return ""

def align_by_id(
    pred_rows: list,
    ref_rows: list,
    image_root: str = "",
    id_key_pred: str = None, 
    id_key_ref:  str = "image_id",
    cap_key_pred: str = "prediction",
    caps_key_ref: str = None
) -> Tuple[list, list, list, list]:
    ref_map = {}
    if caps_key_ref is None and len(ref_rows) > 0:
        if "captions" in ref_rows[0]:
            caps_key_ref = "captions"
        elif "references" in ref_rows[0]:
            caps_key_ref = "references"
        elif "refs" in ref_rows[0]:
            caps_key_ref = "refs"
        else:
            raise KeyError("Nie znaleziono pola z listą referencji w rekordach referencyjnych.")
    for r in ref_rows:
        kid = _norm_img_id(r.get(id_key_ref, r.get("id", "")))
        caps = r.get(caps_key_ref, [])
        if kid and caps:
            ref_map[kid] = [str(c).strip() for c in caps if isinstance(c, str)]

    if id_key_pred is None and len(pred_rows) > 0:
        id_key_pred = "image_id" if "image_id" in pred_rows[0] else "id"

    preds, refs, img_paths, ids = [], [], [], []
    missing_ref, missing_img = 0, 0

    for pr in pred_rows:
        kid = _norm_img_id(pr.get(id_key_pred, pr.get("id", "")))
        if not kid:
            continue
        if kid not in ref_map:
            missing_ref += 1
            continue
        hyp = pr.get(cap_key_pred, "")
        if not isinstance(hyp, str):
            hyp = ""
        preds.append(hyp.strip())
        refs.append(ref_map[kid])
        ids.append(kid)

        p = _id_to_path(kid, image_root) if image_root else ""
        if not p:
            missing_img += 1
        img_paths.append(p)

    print(f"[align_by_id] used={len(preds)}  missing_ref={missing_ref}  missing_img={missing_img}")
    return preds, refs, img_paths, ids


def normalize_lists(predictions, references):
    n = min(len(predictions), len(references))
    preds = [p.strip() if isinstance(p, str) else "" for p in predictions[:n]]
    refs = []
    for rset in references[:n]:
        if not rset:
            refs.append([""])
        else:
            refs.append([str(x).strip() for x in rset if isinstance(x, str)] or [""])
    return preds, refs

def as_coco_dict(preds, refs):
    gts, res = {}, {}
    for i, (p, rset) in enumerate(zip(preds, refs)):
        gg = []
        if rset:
            for s in rset:
                ss = (s if isinstance(s, str) else "")
                gg.append({"caption": ss.strip()})
        else:
            gg.append({"caption": ""})
        gts[i] = gg

        pp = (p if isinstance(p, str) else "")
        res[i] = [{"caption": pp.strip()}]
    return gts, res


def eval_coco_metrics(preds, refs, fast=False):
    out = {}
    gts, res = as_coco_dict(preds, refs)

    gts_ids = set(gts.keys())
    res_ids = set(res.keys())
    inter = gts_ids & res_ids
    print(f"[METR] counts: gts={len(gts_ids)} res={len(res_ids)} inter={len(inter)} missing_in_gts={len(res_ids-gts_ids)} missing_in_res={len(gts_ids-res_ids)}")
    assert len(inter) > 0

    gts = tokenizer.tokenize(gts); res = tokenizer.tokenize(res)
    try:
        print("[METR] BLEU start"); from pycocoevalcap.bleu.bleu import Bleu
        bleu, _ = Bleu(n=4).compute_score(gts, res)
        for i,s in enumerate(bleu,1): out[f"Bleu_{i}"]=float(s)
        print("[METR] BLEU done")

        print("[METR] ROUGE start"); from pycocoevalcap.rouge.rouge import Rouge
        rougeL, _ = Rouge().compute_score(gts, res); out["ROUGE_L"]=float(rougeL)
        print("[METR] ROUGE done")

        print("[METR] CIDEr start"); from pycocoevalcap.cider.cider import Cider
        cider, cider_per_image = Cider().compute_score(gts, res); out["CIDEr"]=float(cider)
        import numpy as np
        print(f"[METR][CIDEr] mean={cider:.6f} sum={np.sum(cider_per_image):.6f} "
        f"min={np.min(cider_per_image):.6f} max={np.max(cider_per_image):.6f} "
        f"n={len(cider_per_image)}")
        print("[METR] CIDEr done")

        if not fast:
            print("[METR] METEOR start"); from pycocoevalcap.meteor.meteor import Meteor
            meteor, _ = Meteor().compute_score(gts, res); out["METEOR"]=float(meteor)
            print("[METR] METEOR done")

            print("[METR] SPICE start"); from pycocoevalcap.spice.spice import Spice
            spice, _ = Spice().compute_score(gts, res)
            out["SPICE"] = float(spice) if isinstance(spice,(int,float)) else \
                sum(float(d["All"]["f"]) for d in spice if "All" in d)/len(spice)
            print("[METR] SPICE done")
    except Exception as e:
        print("[METR] crash:", repr(e))
    return out

def eval_sacrebleu(preds, refs, tokenize=True):
    out = {}
    try:
        print("[METR] SacreBLEU start"); import sacrebleu
        max_k = max(len(r) for r in refs) if refs else 0
        ref_sets = []
        for k in range(max_k):
            ref_sets.append([(r[k] if k < len(r) else r[-1]) for r in refs])
        try:
            if tokenize:
                bleu = sacrebleu.corpus_bleu(preds, ref_sets, tokenize="intl")
            else:
                bleu = sacrebleu.corpus_bleu(preds, ref_sets)
                
            out["SacreBLEU"] = float(bleu.score)
            print("[METR] SacreBLEU done")
        except Exception:
            pass
        try: 
            print("[METR] chrF++ start")
            chrf = sacrebleu.corpus_chrf(preds, ref_sets) 
            out["chrF++"] = float(chrf.score)
            print("[METR] chrF++ done")
        except Exception: 
            pass
    except Exception:
        pass
    return out

def basic_lengths(preds):
    try:
        import numpy as np
        print("[INFO] basic_lengths start")
        lens = [len(p.split()) for p in preds]
        return {
            "Len_pred_tokens_avg": float(np.mean(lens)) if lens else 0.0,
            "Len_pred_tokens_std": float(np.std(lens)) if lens else 0.0,
        }
    except Exception:
        lens = [len(p.split()) for p in preds]
        n = len(lens)
        avg = sum(lens) / n if n else 0.0
        var = sum((x - avg) ** 2 for x in lens) / n if n else 0.0
        print("[INFO] basic_lengths exception")
        return {
            "Len_pred_tokens_avg": float(avg),
            "Len_pred_tokens_std": float(math.sqrt(var)),
        }

class MetricComputer:
    def __init__(
        self,
        bert_model_type: str = "xlm-roberta-large",
        bert_lang: Optional[str] = "pl",
        bert_rescale_with_baseline: bool = False,
        bert_idf: bool = False,
        bert_device: Optional[str] = None,
        clip_model_name: str = "xlm-roberta-base-ViT-B-32",
        clip_pretrained: str = "laion5b_s13b_b90k",
        clip_device: Optional[str] = None,
        clip_bs: int = 16,
        clip_prompt: str = "Na zdjęciu widać ",  # equivalent of "A photo shows ..." loosely translated
    ):
        self.bertscorer = None
        try:
            from bert_score import BERTScorer
            import torch

            if bert_device is None:
                bert_device = "cuda" if torch.cuda.is_available() else "cpu"

            bert_kwargs = dict(
                model_type=bert_model_type,
                idf=bert_idf,
                rescale_with_baseline=bert_rescale_with_baseline,
                device=bert_device,
                batch_size=16
            )
            if bert_lang is not None:
                bert_kwargs["lang"] = bert_lang

            self.bertscorer = BERTScorer(**bert_kwargs)
        except Exception as e:
            print(f"[MetricComputer] Warning: BERTScorer init failed: {e}")
        
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clip_bs = clip_bs
        self.clip_prompt = clip_prompt

        try:
            import torch
            import open_clip

            if clip_device is None:
                clip_device = "cuda" if torch.cuda.is_available() else "cpu"

            self.clip_device = torch.device(clip_device)

            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    clip_model_name, pretrained=clip_pretrained, device=self.clip_device
                )
                tokenizer = open_clip.get_tokenizer(clip_model_name)
            except Exception:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.clip_device
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")

            model.eval()
            self.clip_model = model
            self.clip_preprocess = preprocess
            self.clip_tokenizer = tokenizer
        except Exception as e:
            print(f"[MetricComputer] Warning: OpenCLIP init failed: {e}")
            self.clip_device = None
    
    def bleu1_single(self, pred, refs):
        from pycocoevalcap.bleu.bleu import Bleu
        gts, res = as_coco_dict([pred], [refs])
        gts = tokenizer.tokenize(gts); res = tokenizer.tokenize(res)
        bleu, _ = Bleu(n=4).compute_score(gts, res)
        return bleu[0]
    
    def bleu4_single(self, pred, refs):
        from pycocoevalcap.bleu.bleu import Bleu
        gts, res = as_coco_dict([pred], [refs])
        gts = tokenizer.tokenize(gts); res = tokenizer.tokenize(res)
        bleu, _ = Bleu(n=4).compute_score(gts, res)
        return bleu[3]

    def clipscore_single(self, image_path: str, text: str) -> Dict[str, float]:
        import torch
        from PIL import Image

        im = Image.open(image_path).convert("RGB")
        img = self.clip_preprocess(im).unsqueeze(0).to(self.clip_device)

        prompted = (self.clip_prompt or "") + text
        tokens = self.clip_tokenizer([prompted]).to(self.clip_device)

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            txt_feat = self.clip_model.encode_text(tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            sim = (img_feat * txt_feat).sum(dim=-1).clamp(min=0.0)
            score_100 = float((100.0 * sim).item())

        return score_100


    def compute_metrics(
        self,
        predictions: Sequence[(str)],
        references: Sequence[Union[str, Sequence[str]]],
        predictions_lemmatized: Sequence[(str)] = None,
        references_lemmatized: Sequence[Union[str, Sequence[str]]] = None,
        image_paths_for_clip: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        preds, refs = normalize_lists(predictions, references)
        results: Dict[str, Any] = {}
        
        if predictions_lemmatized is not None and references_lemmatized is not None:
            preds_l, refs_l = normalize_lists(predictions_lemmatized, references_lemmatized)
            results.update(eval_coco_metrics(preds_l, refs_l, False))
            results.update(eval_sacrebleu(preds_l, refs_l, False))
        else:
            results.update(eval_coco_metrics(preds, refs))
            results.update(eval_sacrebleu(preds, refs))

        results.update(self._eval_bertscore_cached(preds, refs))

        if image_paths_for_clip:
            results.update(self._eval_clipscore_cached(image_paths_for_clip, preds))
        else:
            print("No image paths, skipping CLIPScore")

        results.update(basic_lengths(preds))
        print("[INFO] basic_lengths end")
        results["N_samples"] = float(len(preds))
        return results

    def compute_metrics_fast(
        self,
        predictions: Sequence[str],
        references: Sequence[Union[str, Sequence[str]]],
        image_paths_for_clip: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        preds, refs = normalize_lists(predictions, references)
        results: Dict[str, Any] = {}

        results.update(eval_coco_metrics(preds, refs, fast=True))

        results.update(eval_sacrebleu(preds, refs))

        results.update(self._eval_bertscore_cached(preds, refs))

        if image_paths_for_clip:
            results.update(self._eval_clipscore_cached(image_paths_for_clip, preds))
        else:
            print("No image paths, skipping CLIPScore")

        results.update(basic_lengths(preds))
        results["N_samples"] = float(len(preds))
        return results
    
    def compute_metrics_by_id(
        self,
        pred_rows: Sequence[Dict[str, Any]],
        ref_rows: Sequence[Dict[str, Any]],
        image_root: Optional[str] = None,
        id_key_pred: Optional[str] = None,
        id_key_ref: str = "image_id",
        cap_key_pred: str = "prediction",
        caps_key_ref: Optional[str] = None,
        fast: bool = True,
    ) -> Dict[str, Any]:
        preds, refs, img_paths, ids = align_by_id(
            list(pred_rows),
            list(ref_rows),
            image_root or "",
            id_key_pred=id_key_pred,
            id_key_ref=id_key_ref,
            cap_key_pred=cap_key_pred,
            caps_key_ref=caps_key_ref,
        )
        if not preds:
            raise RuntimeError("Przecięcie między predykcjami a referencjami to zbiór pusty!!")

        if fast:
            results = self.compute_metrics_fast(preds, refs, img_paths if image_root else None)
        else:
            results = self.compute_metrics(preds, refs, None, None, img_paths if image_root else None)

        results["_ids_count"] = len(ids)
        results["_missing_img_paths"] = sum(1 for p in img_paths if not p)
        return results


    def _eval_bertscore_cached(self, preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.bertscorer is None or not preds:
            return out

        try:
            print("[METR] BERTScore start"); import numpy as np
            max_k = max(len(r) for r in refs) if refs else 0
            if max_k == 0:
                return out

            best_f1 = None
            for k in range(max_k):
                col = [(r[k] if k < len(r) else r[-1]) for r in refs]
                P, R, F1 = self.bertscorer.score(preds, col)
                f1 = F1.detach().cpu().numpy()
                if best_f1 is None:
                    best_f1 = f1
                else:
                    best_f1 = np.maximum(best_f1, f1)

            if best_f1 is not None:
                out["BERTScore_F1"] = float(best_f1.mean())
                print("[METR] BERTScore end")
        except Exception as e:
            print(f"[MetricComputer] BERTScore failed: {e}")
        return out

    def _eval_clipscore_cached(self, image_paths: Sequence[str], texts: Sequence[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.clip_model is None or self.clip_preprocess is None or self.clip_tokenizer is None:
            return out
        try:
            import torch
            import numpy as np
            from PIL import Image
            print("[METR] CLIPScore start");
            n = min(len(image_paths), len(texts))
            if n == 0:
                return out

            image_paths = image_paths[:n]
            texts = texts[:n]

            prompted_texts = [self.clip_prompt + t for t in texts]

            img_feats = []
            bs = max(1, self.clip_bs)
            for i in range(0, n, bs):
                batch_paths = image_paths[i:i + bs]
                imgs = []
                for p in batch_paths:
                    im = Image.open(p).convert("RGB")
                    imgs.append(self.clip_preprocess(im))
                imgs = torch.stack(imgs).to(self.clip_device)
                with torch.no_grad():
                    feats = self.clip_model.encode_image(imgs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                img_feats.append(feats)
            img_feats = torch.cat(img_feats, dim=0)

            tokens = self.clip_tokenizer(prompted_texts).to(self.clip_device)
            with torch.no_grad():
                txt_feats = self.clip_model.encode_text(tokens)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            sims = (img_feats * txt_feats).sum(dim=-1).clamp(min=0)
            scores_2_5 = 2.5 * sims
            scores_100 = 100.0 * sims

            score_2_5_np = scores_2_5.detach().cpu().numpy()
            scores_100_np = scores_100.detach().cpu().numpy()

            out["CLIPScore_mean_100"] = float(np.mean(scores_100_np))
            out["CLIPScore_std_100"] = float(np.std(scores_100_np))

            out["CLIPScore_mean_2_5"] = float(np.mean(score_2_5_np))
            out["CLIPScore_std_2_5"] = float(np.std(score_2_5_np))

            print("[METR] CLIPcore end")
        except Exception as e:
            print(f"[MetricComputer] CLIPScore failed: {e}")
        return out
