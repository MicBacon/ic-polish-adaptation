import unicodedata
import re, jsonlines, json

INPUT_FILES = [#'../Qwen2_5-VL/eval_results/raw_pl_test_hq/predictions_nb_e1_128.jsonl',
               #'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions_nb_e1_128.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_pl_test_hq/predictions.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions.jsonl',
              #'translate_evaluate_results/bing/Qwen2_5-VL_raw_en_test_hq.json',
              #'translate_evaluate_results/bing/Qwen2_5-VL_raw_en_test_std.json',
              '../Qwen2_5-VL/eval_results/raw_pl_test_hq/predictions_nb_e1_128_update.jsonl'
              ]

def nfc(s):
    return unicodedata.normalize("NFC", s)

POLISH_LETTERS = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż" \
                 "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ" \
                 "qvxQVX"
DIGITS = "0123456789"

PUNCT = (
    ".,;:!?…"
    "--–—"
    "\"'„”‚‘’«»"
    "()[]{}"
    "/\\%&@#*+=_<>\u007C^"
)

WHITESPACE = " \t\r\n\u00A0\u202F"

ALLOWED = set(POLISH_LETTERS + DIGITS + PUNCT + WHITESPACE)

def is_char_ok(ch: str) -> bool:
    ch = nfc(ch)
    return ch in ALLOWED

sus_candidates = []
for input_file in INPUT_FILES:
    if(input_file.endswith('.jsonl')):
        with jsonlines.open(input_file) as data:
            for line in data:
                pred = line.get('prediction')
                img_id = line.get('id')
                for char in pred:
                    if not is_char_ok(char):
                        sus_candidates.append((img_id, pred))
                        break
    else:
        with open(input_file) as input:
            data = json.load(input)
            sus_candidates = [
                (r.get("img_id"), r.get("pl_prediction", ""))
                for r in data
                    if isinstance(r, dict)
                    and "pl_prediction" in r
                    and any(not is_char_ok(ch) for ch in r.get("pl_prediction", ""))
            ]


print(sus_candidates, 'Liczba referencji z podejrzanymi znakami: ', len(sus_candidates))