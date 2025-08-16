import json
import translators as ts
from bert_score import score
import os
import nltk

with open('output/coco_caption_large/result/vqa_result_epoch10_rank0.json', 'r') as f:
    data = json.load(f)
    #_ = ts.preaccelerate_and_speedtest()

    # new json to store results
    os.makedirs(os.path.dirname("output/coco_caption_large/result/vqa_result_epoch10_rank0_translated_captions.json"),
                 exist_ok=True)

    r = open("output/coco_caption_large/result/vqa_result_epoch10_rank0_translated_captions.json", "w")

    for item in data[:25]:
       caption = item['pred_caption']
       caption_pl = ts.translate_text(caption, translator='google', to_language='pl')
       #BLEUscore = nltk.translate.bleu_score.sentence_bleu(caption.split(), caption_pl.split())

       P, R, F1 = score(
            [caption_pl],
            [caption],
            lang="multilingual"
        )
       
       json.dump({"question_id": item['question_id'], "pred_caption": caption, 
                           "translated_caption": caption_pl, #"BLEU_score": BLEUscore,
                           "bert_score_precision": P.numpy().tolist(),
                           "bert_score_recall": R.numpy().tolist(),
                           "bert_score_f1": F1.numpy().tolist()}, 
                           r, ensure_ascii=False)
       r.write('\n')

    r.close()

    # print('Translating to Polish...')
    # caption = data[0]['pred_caption']
    # caption_pl = ts.translate_text(caption, translator='google', to_language='pl')
    # print(caption_pl)