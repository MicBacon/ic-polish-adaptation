import json, jsonlines
import translators as ts
import os, csv, re
import time
import random
import stanza
from bert_score import BERTScorer
from MetricComputer import MetricComputer

INPUT_PATH = '../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions.jsonl'
RESULT_PATH = 'translate_evaluate_results/'
DST_FILE = 'Qwen2_5-VL_raw_en_test_std.json'

TRANSLATOR = 'bing'
PREACCELERATE = False
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
            print(e)
            time.sleep(sleep_s)
    raise RuntimeError(f"failed after {max_retries} retries ({translator}): {last_exc}")

if PREACCELERATE:
    _ = ts.preaccelerate_and_speedtest()

#print(translate_with_retry('a group of people riding skateboards down a street', translator=TRANSLATOR))
#mc = MetricComputer()
#print(mc.compute_metrics([], [], None, None, []))

with jsonlines.open(INPUT_PATH) as i:
    for line in i:
        pred = line.get('prediction', '')
        references_pl = line.get('references', [])
        print(lemmatize_pl(pred))
        print([lemmatize_pl(ref) for ref in references_pl])
