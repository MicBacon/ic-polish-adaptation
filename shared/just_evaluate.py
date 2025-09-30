import json, jsonlines
import os, re, stanza
from MetricComputer import MetricComputer

INPUT_PATHS = [#'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions_nb_e1_128.jsonl',
               #'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions.jsonl',
                #'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions_nb_e2_512.jsonl'
                '../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions_nb_e1_128_update.jsonl'
               ]
RESULT_PATH = 'just_evaluate_results/'
DST_FILES = [#'Qwen2_5-VL_raw_pl_test_std_nb_e1_128.json',
             #'Qwen2_5-VL_raw_pl_test_std.json',
             #'Qwen2_5-VL_raw_pl_test_std_nb_e2_512.json',
             'Qwen2_5-VL_raw_pl_test_std_nb_e1_128_update.json',
             ]

nlp = stanza.Pipeline(
    lang="pl",
    processors="tokenize,lemma",
    tokenize_no_ssplit=True,
    use_gpu=False
)

_ws = re.compile(r"\s+")

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

mc = MetricComputer()
for input_path, dst_file in zip(INPUT_PATHS, DST_FILES):
    if(input_path.endswith('jsonl')):
        data = jsonlines.open(input_path)
        
        os.makedirs(RESULT_PATH, exist_ok=True)
        r = open(os.path.join(RESULT_PATH, dst_file), "w")
        r.write('[\n')

        all_predictions = []
        all_predictions_l = []
        all_references = []
        all_references_l = []
        all_img_paths = []

        for line in data:
            pred = line['prediction']
            pred_l = lemmatize_pl(line['prediction'])
            references = line['references']
            references_l = [lemmatize_pl(r) for r in line['references']]
            img_path = '/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/shared/data/flickr30k/Images/' + line['id'] + '.jpg'
                    
            all_predictions.append(pred)
            all_predictions_l.append(pred_l)
            all_references.append(references)
            all_references_l.append(references_l)
            all_img_paths.append(img_path)

    metrics = mc.compute_metrics(all_predictions, all_references, all_predictions_l, all_references_l, all_img_paths)
    print(metrics)
    json.dump({"overall_metric_score": metrics}, r, ensure_ascii=False)
    r.write('\n]')
    r.close()