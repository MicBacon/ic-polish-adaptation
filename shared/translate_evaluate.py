import json
import translators as ts
import os
import csv
from bert_score import BERTScorer
import MetricComputer as cm

CAPTION_COUNT = 3783
RESULT_PATH = 'output/coco_caption_large/result/'
TRANSLATED_RESULT_PATH = 'output/coco_caption_large/result_translated/'
SRC_FILE = 'vqa_result_flickr30k_epoch5.json'
DST_FILE = 'vqa_result_flickr30k_epoch5_translated.json'
POLISH_DESC_PATH = '../shared/data/flickr30k/descriptions_flickr30k_translated.csv'

_ = ts.preaccelerate_and_speedtest()

# separate BERTscorer for machine translation
bs = BERTScorer(model_type="xlm-roberta-large", rescale_with_baseline=False)

with open(os.path.join(RESULT_PATH, SRC_FILE), 'r') as f:
    data = json.load(f)

    os.makedirs(TRANSLATED_RESULT_PATH, exist_ok=True)

    r = open(os.path.join(TRANSLATED_RESULT_PATH, DST_FILE), "w")
    r.write('[')
    
    for idx, item in enumerate(data[:CAPTION_COUNT]):
       caption = item['pred_caption']

       # polish supported:
       # google ->  support more languages in the world
       # yandex -> support more languages in the world, support word to emoji
       # bing -> support more languages in the world
       # baidu -> support most languages, support professional field, support classical Chinese
       # sogou -> support more languages in the world
       # deepl -> high quality to translate but response slowly
       # argos -> open-source

       if(caption == ""):
           caption = "a photo"
           print("found empty caption: \n", item)

       caption_pl = ts.translate_text(caption, translator='google', to_language='pl')
       

       # calculate multilingual translation bert scores
       P, R, F1 = bs.score([caption], [caption_pl])
       
       # get polish descriptions
       captions_pl = []
       with open(POLISH_DESC_PATH, "r") as desc_file:
           desc_reader = csv.reader(desc_file, delimiter='|')
           for row in desc_reader:
               if row[0] == item['question_id'].split('.')[0]:
                   captions_pl.append(row[1])

       # check if polish references exist
       if(len(captions_pl) > 0):
           # calculate image captioning metrics

           #captioning_metrics = cm.compute_metrics([caption_pl], [captions_pl], None)

           json.dump({"question_id": item['question_id'], "pred_caption": caption, "translated_caption": caption_pl,
                    "translation_scores":{"P":P.numpy().tolist(), "R":R.numpy().tolist(), "F1":F1.numpy().tolist()},
                    "polish_references": captions_pl,
                    #"captioning_metrics": captioning_metrics
                    },
                    r, ensure_ascii=False)
        
           if ((idx+1) % 100 == 0):
            print(f"Processed {idx+1} captions.")

           if idx < CAPTION_COUNT - 1:
            r.write(',\n')
       else:
          CAPTION_COUNT = CAPTION_COUNT-1
          print('Found 0 references for', item['question_id'])

    r.write('\n]')
    r.close()
    
with open(os.path.join(TRANSLATED_RESULT_PATH, DST_FILE), "r") as f:
    data = json.load(f)
    hypotheses = []
    references = []
    img_paths = []
    for item in data:
        hypotheses.append(item['translated_caption'])
        references.append(item['polish_references'])
        img_paths.append("/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/shared/data/flickr30k/Images/" + item['question_id'])
    scores = cm.compute_metrics(hypotheses, references, img_paths)
    print(scores)
