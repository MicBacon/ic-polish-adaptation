from typing import List, Sequence, Optional, Dict, Any, Union
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

class MetricComputer:

    def __init__(
        self,
        bert_model_type: str = "xlm-roberta-large",
        bert_lang: Optional[str] = None,
        bert_rescale_with_baseline: bool = False,
        bert_idf: bool = True,
        bert_device: Optional[str] = None,
        clip_model_name: str = "xlm-roberta-base-ViT-B-32",
        clip_pretrained: str = "laion5b_s13b_b90k",
        clip_device: Optional[str] = None,
        clip_bs: int = 16,
        clip_prompt_pl: str = "Na zdjęciu widać ",  # "A photo shows ..." luźno przetłumaczone
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
        self.clip_prompt_pl = clip_prompt_pl

        try:
            import torch
            import open_clip

            if clip_device is None:
                clip_device = "cuda" if torch.cuda.is_available() else "cpu"

            self.clip_device = torch.device(clip_device)

            try:
                # Try user-specified model first
                model, _, preprocess = open_clip.create_model_and_transforms(
                    clip_model_name, pretrained=clip_pretrained, device=self.clip_device
                )
                tokenizer = open_clip.get_tokenizer(clip_model_name)
            except Exception:
                # Fallback to a widely available config from docs
                # (ViT-B/32 + laion2b_s34b_b79k)
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

    def compute_metrics(
        self,
        predictions: Sequence[str],
        references: Sequence[Union[str, Sequence[str]]],
        image_paths_for_clip: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        preds, refs = normalize_lists(predictions, references)
        results: Dict[str, Any] = {}

        results.update(eval_coco_metrics(preds, refs, image_paths_for_clip))

        results.update(eval_sacrebleu(preds, refs))

        results.update(self._eval_bertscore_cached(preds, refs))

        if image_paths_for_clip:
            results.update(self._eval_clipscore_cached(image_paths_for_clip, preds))
        else:
            print("No image paths, skipping CLIPScore")

        results.update(basic_lengths(preds))
        results["N_samples"] = float(len(preds))
        return results


    def _eval_bertscore_cached(self, preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.bertscorer is None or not preds:
            return out

        try:
            import numpy as np
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

            n = min(len(image_paths), len(texts))
            if n == 0:
                return out

            image_paths = image_paths[:n]
            texts = texts[:n]

            prompted_texts = [self.clip_prompt_pl + t for t in texts]

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
            scores = 100.0 * sims
            scores_np = scores.detach().cpu().numpy()

            out["CLIPScore_mean"] = float(np.mean(scores_np))
            out["CLIPScore_std"] = float(np.std(scores_np))
        except Exception as e:
            print(f"[MetricComputer] CLIPScore failed: {e}")
        return out
