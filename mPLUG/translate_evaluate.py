import json
import translators as ts
import os
import csv
from bert_score import score
import compute_metrics

CAPTION_COUNT = 1000
RESULT_PATH = 'output/coco_caption_large/result/'

_ = ts.preaccelerate_and_speedtest()

with open(os.path.join(RESULT_PATH, 'vqa_result_flickr30k_epoch4.json'), 'r') as f:
    data = json.load(f)

    os.makedirs("output/coco_caption_large/result_translated", exist_ok=True)

    r = open("output/coco_caption_large/result_translated/vqa_result_epoch10_rank0_translated_captions.json", "w")
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

       caption_pl = ts.translate_text(caption, translator='deepl', to_language='pl')

       # calculate multilingual translation bert scores
       P, R, F1 = score(
            [caption_pl],
            [caption],
            lang="multilingual"
       )

       # get polish descriptions
       captions_pl = []
       with open("../shared/data/flickr30k/descriptions_flickr30k_translated.csv", "r") as desc_file:
           desc_reader = csv.reader(desc_file, delimiter='|')
           for row in desc_reader:
               if row[0] == item['question_id'].split('.')[0]:
                   captions_pl.append(row[1])

       # check if polish references exist
       if(len(captions_pl) > 0):
           # calculate image captioning metrics
           captioning_metrics = compute_metrics.compute_metrics(captions_pl, caption_pl, image_paths=None)

           json.dump({"question_id": item['question_id'], "pred_caption": caption, "translated_caption": caption_pl,
                    "translation_scores":{"P":P.numpy().tolist(), "R":R.numpy().tolist(), "F1":F1.numpy().tolist()},
                    "polish_references": captions_pl,
                    "captioning_metrics": captioning_metrics},
                    r, ensure_ascii=False)
        
           if (idx % 100 == 0):
            print(f"Processed {idx} captions.")

           if idx < CAPTION_COUNT - 1:
            r.write(',\n')

    r.write('\n],')
    r.write('[')

    
    with open("output/coco_caption_large/result_translated/vqa_result_epoch10_rank0_translated_captions.json", "r") as f:
        data = json.load(f)
        references = []
        img_paths = []
        hypotheses = []
        for item in data:
            references.append(item['polish_references'])
            hypotheses.append(item['translated_caption'])
            img_paths.append("Images/" + item['question_id'])
        scores = compute_metrics.compute_metrics(references, hypotheses, img_paths)
        json.dump({"overall_scores": scores}, r, ensure_ascii=False)
        print(scores)
    r.write(']')
    r.close()