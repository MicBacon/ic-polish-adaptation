import json

EN_ALL = 'flickr30k_captions_all.json'
PL_SPLITS = [
    ('flickr30kPolish_captions_train.json', 'flickr30kEnglish_captions_train.json'),
    ('flickr30kPolish_captions_val.json', 'flickr30kEnglish_captions_val.json'),
    ('flickr30kPolish_captions_test_std.json', 'flickr30kEnglish_captions_test_std.json'),
    ('flickr30kPolish_captions_test_hq.json', 'flickr30kEnglish_captions_test_hq.json'),
]

with open(EN_ALL, 'r', encoding='utf-8') as f:
    en_data = json.load(f)

en_map = {}
for rec in en_data:
    k = str(rec.get('image_id', '').split('.')[0])
    caps = rec.get('captions') or rec.get('references') or []
    if k:
        en_map[k] = caps

def build(split_path):
    with open(split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for rec in data:
        k = str(rec.get('image_id', ''))
        out.append({'image_id': k, 'captions': en_map.get(k, [])})
    return out

for src, dst in PL_SPLITS:
    out = build(src)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
