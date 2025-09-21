from typing import List, Dict, Any, Tuple
import math

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
    """
    Format wymagany przez pycocoevalcap:
      - gts: {id: [ref1, ref2, ...]}
      - res: {id: [hyp]}
    """
    gts, res = {}, {}
    for i, (p, rset) in enumerate(zip(preds, refs)):
        gts[i] = rset
        res[i] = [p]
    return gts, res

def _eval_coco_metrics(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        # BLEU
        try:
            from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
            gts, res = _as_coco_dict(preds, refs)
            scorer = Bleu(n=4)
            score, _ = scorer.compute_score(gts, res)  # score: list[4]
            # Zwroty w stylu COCOEvalCap:
            for i, s in enumerate(score, start=1):
                out[f"Bleu_{i}"] = float(s)
        except Exception:
            pass

        # METEOR
        try:
            from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
            gts, res = _as_coco_dict(preds, refs)
            scorer = Meteor()
            score, _ = scorer.compute_score(gts, res)
            out["METEOR"] = float(score)
        except Exception:
            pass

        # ROUGE_L
        try:
            from pycocoevalcap.rouge.rouge import Rouge  # type: ignore
            gts, res = _as_coco_dict(preds, refs)
            scorer = Rouge()
            score, _ = scorer.compute_score(gts, res)
            out["ROUGE_L"] = float(score)
        except Exception:
            pass

        # CIDEr
        try:
            from pycocoevalcap.cider.cider import Cider  # type: ignore
            gts, res = _as_coco_dict(preds, refs)
            scorer = Cider()
            score, _ = scorer.compute_score(gts, res)
            out["CIDEr"] = float(score)
        except Exception:
            pass

        # SPICE (opcjonalnie; wymaga Java)
        try:
            from pycocoevalcap.spice.spice import Spice  # type: ignore
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
        # Cały blok COCO nie powiódł się — zwracamy to, co mamy
        pass
    return out

def _eval_sacrebleu(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    """
    SacreBLEU i chrF++ (skala 0..100). Tokenizacja 'intl' jest bezpieczna dla języków z diakrytykami.
    """
    out: Dict[str, float] = {}
    try:
        import sacrebleu  # type: ignore

        # Zbuduj ref_sets w układzie: [refset1, refset2, ...] gdzie każdy refset ma długość = N
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
            # API v2: sacrebleu.metrics.chrf.CHRF
            try:
                from sacrebleu.metrics import CHRF  # type: ignore
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
    """
    BERTScore F1 (lang='pl'), best-of po wielu referencjach.
    Zwraca średnią F1 (0..1).
    """
    out: Dict[str, float] = {}
    try:
        from bert_score import score as bert_score  # type: ignore
        import numpy as np  # type: ignore

        if not preds:
            return out

        # Przygotuj kolumny referencji (wyrównanie braków ostatnią dostępną)
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
        import numpy as np  # type: ignore
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

def compute_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Główna funkcja wołana przez skrypt ewaluacyjny.
    """
    preds, refs = _normalize_lists(predictions, references)

    results: Dict[str, float] = {}
    # COCO metrics (Bleu_1..4, METEOR, ROUGE_L, CIDEr, SPICE*)
    results.update(_eval_coco_metrics(preds, refs))

    # SacreBLEU / chrF++
    results.update(_eval_sacrebleu(preds, refs))

    # BERTScore (PL, best-of)
    results.update(_eval_bertscore(preds, refs))

    # Długości (pomocnicze)
    results.update(_basic_lengths(preds))

    # Informacyjnie: ile przykładów
    results["N_samples"] = float(len(preds))

    return results