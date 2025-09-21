import math

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
        gts[i] = rset
        res[i] = [p]
    return gts, res

def eval_coco_metrics(preds, refs, img_paths):
    out = {}
    try:
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            gts, res = as_coco_dict(preds, refs)
            scorer = Bleu(n=4)
            score, _ = scorer.compute_score(gts, res)
            for i, s in enumerate(score, start=1):
                out[f"Bleu_{i}"] = float(s)
        except Exception:
            pass
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            gts, res = as_coco_dict(preds, refs)
            scorer = Meteor()
            score, _ = scorer.compute_score(gts, res)
            out["METEOR"] = float(score)
        except Exception:
            pass
        try:
            from pycocoevalcap.rouge.rouge import Rouge
            gts, res = as_coco_dict(preds, refs)
            scorer = Rouge()
            score, _ = scorer.compute_score(gts, res)
            out["ROUGE_L"] = float(score)
        except Exception:
            pass
        try:
            from pycocoevalcap.cider.cider import Cider
            gts, res = as_coco_dict(preds, refs)
            scorer = Cider()
            score, _ = scorer.compute_score(gts, res)
            out["CIDEr"] = float(score)
        except Exception:
            pass
        try:
            from pycocoevalcap.spice.spice import Spice
            gts, res = as_coco_dict(preds, refs)
            scorer = Spice()
            score, _ = scorer.compute_score(gts, res)
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

def eval_sacrebleu(preds, refs):
    out = {}
    try:
        import sacrebleu
        max_k = max(len(r) for r in refs) if refs else 0
        ref_sets = []
        for k in range(max_k):
            ref_sets.append([(r[k] if k < len(r) else r[-1]) for r in refs])
        try:
            bleu = sacrebleu.corpus_bleu(preds, ref_sets, tokenize="intl")
            out["SacreBLEU"] = float(bleu.score)
        except Exception:
            pass
    except Exception:
        pass
    return out

def eval_bertscore(preds, refs):
    out = {}
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
            col = [(r[k] if k < len(r) else r[-1]) for r in refs]
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

def basic_lengths(preds):
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

def eval_clipscore_pl(image_paths, texts, clip_model_name="xlm-roberta-base-ViT-B-32", clip_pretrained="laion5b_s13b_b90k", clip_bs=16):
    out = {}
    try:
        import torch
        import open_clip
        import numpy as np
        from PIL import Image

        if not image_paths or not texts:
            return out

        n = min(len(image_paths), len(texts))
        image_paths = image_paths[:n]
        texts = texts[:n]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=clip_pretrained, 
            device=device
        )
        model.eval()

        prompt = "Na zdjęciu widać " # polski ekwiwalent "A photo of" luźno przetłumaczony przeze mnie 
        prompted_texts = [prompt + text for text in texts]

        img_feats = []
        bs = clip_bs
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

        text_tokens = open_clip.tokenize(prompted_texts).to(device)
        with torch.no_grad():
            txt_feats = model.encode_text(text_tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims = (img_feats * txt_feats).sum(dim=-1).clamp(min=0)
        scores = 100.0 * sims

        scores_np = scores.detach().cpu().numpy()
        out["CLIPScore_mean"] = float(np.mean(scores_np))
        out["CLIPScore_std"] = float(np.std(scores_np))

    except Exception as e:
        print(e)
    return out

def compute_metrics(predictions, references, image_paths_for_metrics):
    preds, refs = normalize_lists(predictions, references)
    results = {}
    results.update(eval_coco_metrics(preds, refs, image_paths_for_metrics))
    results.update(eval_sacrebleu(preds, refs))
    results.update(eval_bertscore(preds, refs))
    if image_paths_for_metrics:
        results.update(eval_clipscore_pl(image_paths=image_paths_for_metrics, texts=preds))
    else:
        print("No image paths, skipping CLIPScore")

    if "CLIPScore" in results and "CIDEr" in results:
        clip_score = results["CLIPScore"]
        cider_score = results["CIDEr"]
        
        # RefCLIPScore = (2 * CLIPScore * CIDEr) / (CLIPScore + CIDEr)
        if clip_score + cider_score > 0:
            results["RefCLIPScore"] = (2 * clip_score * cider_score) / (clip_score + cider_score)
        else:
            results["RefCLIPScore"] = 0.0

    results.update(basic_lengths(preds))
    results["N_samples"] = float(len(preds))
    return results