import json, csv

val_f = open("flickr30kPolish_captions_val.json", "r", encoding="utf-8")
test_f = open("flickr30kPolish_captions_test.json", "r", encoding="utf-8")
train_f = open("flickr30kPolish_captions_train.json", "r", encoding="utf-8")

val_set_AIDe = json.load(val_f)
test_set_AIDe = json.load(test_f)
train_set_flickr = json.load(train_f)

val_f_image_ids = set([item["image_id"].__str__() for item in val_set_AIDe])
test_f_image_ids = set([item["image_id"].__str__() for item in test_set_AIDe])
train_f_image_ids = set([item["image_id"].__str__() for item in train_set_flickr])

print(f"Validation set size: {len(val_f_image_ids)}")
print(f"Test set size: {len(test_f_image_ids)}")
print(f"Training set size: {len(train_f_image_ids)}")

print(f"Training set - validation overlap: {len(train_f_image_ids.intersection(val_f_image_ids))}")
print(f"Training set - test overlap: {len(train_f_image_ids.intersection(test_f_image_ids))}")
print(f"Validation set - test overlap: {len(val_f_image_ids.intersection(test_f_image_ids))}")

img_id_caption_count_train = [(item["image_id"].__str__(), len(item["captions"])) for item in train_set_flickr]
print(f"Training set - images with 7 captions: {len([item for item in img_id_caption_count_train if item[1] == 7])}")

with open("captions.csv", "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    data_AIDe = list(reader)

aide_images = set(item["Picture_orig_name"].split("_")[0] for item in data_AIDe)
print(len(aide_images))

overall_flickr30k_size = csv.reader(open("descriptions_flickr30k_translated.csv", "r"), delimiter='|')
flickr_images = set(row[0] for row in overall_flickr30k_size)
print(len(flickr_images))