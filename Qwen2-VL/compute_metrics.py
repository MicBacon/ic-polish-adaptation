from typing import List, Dict, Any, Tuple
import math

from pyparsing import Optional

def _normalize_lists(predictions: List[str], references: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    n = min(len(predictions), len(references))
    preds = [p.strip() if isinstance(p, str) else "" for p in predictions[:n]]
    refs = []
    for rset in references[:n]:
        if not rset:
            refs.append([""])
        else:
            refs.append([str(x).strip() for x in rset if isinstance(x, str)] or [""])
    return preds, refs

def _as_coco_dict(preds: List[str], refs: List[List[str]]) -> Tuple[Dict[Any, List[str]], Dict[Any, List[str]]]:
    gts, res = {}, {}
    for i, (p, rset) in enumerate(zip(preds, refs)):
        gts[i] = rset
        res[i] = [p]
    return gts, res

def _eval_coco_metrics(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            gts, res = _as_coco_dict(preds, refs)
            scorer = Bleu(n=4)
            score, _ = scorer.compute_score(gts, res)

            for i, s in enumerate(score, start=1):
                out[f"Bleu_{i}"] = float(s)
        except Exception:
            pass

        try:
            from pycocoevalcap.meteor.meteor import Meteor  
            gts, res = _as_coco_dict(preds, refs)
            scorer = Meteor()
            score, _ = scorer.compute_score(gts, res)
            out["METEOR"] = float(score)
        except Exception:
            pass

        try:
            from pycocoevalcap.rouge.rouge import Rouge  
            gts, res = _as_coco_dict(preds, refs)
            scorer = Rouge()
            score, _ = scorer.compute_score(gts, res)
            out["ROUGE_L"] = float(score)
        except Exception:
            pass

        # CIDEr
        try:
            from pycocoevalcap.cider.cider import Cider  
            gts, res = _as_coco_dict(preds, refs)
            scorer = Cider()
            score, _ = scorer.compute_score(gts, res)
            out["CIDEr"] = float(score)
        except Exception:
            pass

        # SPICE (opcjonalnie; wymaga Java)
        try:
            from pycocoevalcap.spice.spice import Spice  
            gts, res = _as_coco_dict(preds, refs)
            scorer = Spice()
            score, _ = scorer.compute_score(gts, res)
            # SPICE zwykle zwraca dict per obraz; w niektórych forku zwraca float średni.
            # W powszechnych implementacjach compute_score zwraca (avg_score, scores)
            # więc jeśli score jest scalarem, bierzemy go; jeśli listą dictów — uśredniamy 'All':
            if isinstance(score, (int, float)):
                out["SPICE"] = float(score)
            else:
                try:
                    vals = []
                    for d in score:
                        if isinstance(d, dict) and "All" in d and "f" in d["All"]:
                            vals.append(float(d["All"]["f"]))
                    if vals:
                        out["SPICE"] = sum(vals) / len(vals)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    return out

def _eval_sacrebleu(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import sacrebleu  

        max_k = max(len(r) for r in refs) if refs else 0
        ref_sets = []
        for k in range(max_k):
            ref_sets.append([ (r[k] if k < len(r) else r[-1]) for r in refs ])

        # BLEU (sacre)
        try:
            bleu = sacrebleu.corpus_bleu(preds, ref_sets, tokenize="intl")
            out["SacreBLEU"] = float(bleu.score)
        except Exception:
            pass

        # chrF++
        try:
            try:
                from sacrebleu.metrics import CHRF  
                chrf = CHRF()
                chrf_res = chrf.corpus_score(preds, ref_sets)
                out["chrF++"] = float(chrf_res.score)
            except Exception:
                # starsze API:
                chrf_res = sacrebleu.corpus_chrf(preds, ref_sets)
                out["chrF++"] = float(chrf_res.score)
        except Exception:
            pass
    except Exception:
        pass
    return out

def _eval_bertscore(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        from bert_score import score as bert_score  
        import numpy as np  

        if not preds:
            return out

        max_k = max(len(r) for r in refs) if refs else 0
        if max_k == 0:
            return out

        best_f1 = None
        for k in range(max_k):
            col = [ (r[k] if k < len(r) else r[-1]) for r in refs ]
            _, _, F1 = bert_score(preds, col, lang="pl", rescale_with_baseline=True)
            f1 = F1.detach().cpu().numpy()
            if best_f1 is None:
                best_f1 = f1
            else:
                best_f1 = np.maximum(best_f1, f1)

        if best_f1 is not None:
            out["BERTScore_F1"] = float(best_f1.mean())
    except Exception:
        pass
    return out

def _basic_lengths(preds: List[str]) -> Dict[str, float]:
    try:
        import numpy as np  
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
        return {
            "Len_pred_tokens_avg": float(avg),
            "Len_pred_tokens_std": float(math.sqrt(var)),
        }
    
def _eval_clipscore(
    image_paths: List[str],
    texts: List[str],
    clip_model_name: str = "ViT-L-14",
    clip_pretrained: str = "openai",
    mclip_model: str = "M-CLIP/LABSE-Vit-L-14",
    clip_bs: int = 16,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import torch
        import open_clip
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from PIL import Image

        if not image_paths or not texts:
            return out
        n = min(len(image_paths), len(texts))
        image_paths = image_paths[:n]
        texts = texts[:n]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=str(device)
        )
        model.eval()

        mclip = SentenceTransformer(mclip_model, device=str(device))

        img_feats = []
        bs = max(1, int(clip_bs))
        for i in range(0, n, bs):
            batch_paths = image_paths[i:i+bs]
            imgs = []
            for p in batch_paths:
                im = Image.open(p).convert("RGB")
                imgs.append(preprocess(im))
            imgs = torch.stack(imgs).to(device)
            with torch.no_grad():
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            img_feats.append(feats)
        img_feats = torch.cat(img_feats, dim=0)

        with torch.no_grad():
            txt_feats = mclip.encode(texts, convert_to_tensor=True, device=str(device), normalize_embeddings=True)

        sims = (img_feats * txt_feats).sum(dim=-1).clamp(min=0).detach().cpu().numpy()
        scores = 100.0 * sims  # skala 0..100

        out["CLIPScore_mean"] = float(np.mean(scores))
        out["CLIPScore_std"] = float(np.std(scores))
    except Exception as e:
        pass
    return out

def compute_metrics(predictions: List[str], references: List[List[str]], image_paths_for_metrics: List[str]) -> Dict[str, float]:
    preds, refs = _normalize_lists(predictions, references)

    results: Dict[str, float] = {}

    results.update(_eval_coco_metrics(preds, refs))

    results.update(_eval_sacrebleu(preds, refs))

    results.update(_eval_bertscore(preds, refs))

    if image_paths_for_metrics:
        results.update(_eval_clipscore(
            image_paths=image_paths_for_metrics,
            texts=preds
        ))

    results.update(_basic_lengths(preds))
    results["N_samples"] = float(len(preds))

    return results