import csv
import os
import json
import random
import pandas as pd

aideOk = False
flickrPolishOk = False
images_folder = "Images"
flickr_folder_images = set(os.path.splitext(filename)[0] for filename in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, filename)))

# first check if all AIDe annnotations are present
with open("captions.csv", "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    data_AIDe = list(reader)

aide_images = set(item["Picture_orig_name"].split("_")[0] for item in data_AIDe)
missing_images = aide_images - flickr_folder_images

if missing_images:
    print("Missing AIDe images:")
    for img in missing_images:
        print(" -", img)
else:
    print("All AIDe images are present in the dataset.")
    aideOk = True

# second check if all Flickr30k annotations are present
with open("descriptions_flickr30k_translated.csv", "r") as f:
    df = pd.read_csv(f, delimiter="|")
    df = df.groupby("image_name")['comment'].apply(list)
    data_Flickr = list(df.items())

flickr_polish_images = set(item[0].__str__() for item in data_Flickr)
missing_images = flickr_polish_images - flickr_folder_images

if missing_images:
    print("Missing polish Flickr30k images:")
    for img in missing_images:
        print(" -", img)
else:
    print("All polish Flickr30k images are present in the dataset.")
    flickrPolishOk = True

if flickrPolishOk and aideOk:
    print("Data is correct. Starting merging process...")
    # split AIDe annotations randomly with fixed seed
    print("Building training, validation and test sets...")
    random.seed(42)

    random.shuffle(data_AIDe)
    random.shuffle(data_Flickr)

    aide_unique = {}
    for item in data_AIDe:
        img_id = item["Picture_orig_name"].split("_")[0]
        if img_id not in aide_unique and img_id in flickr_folder_images:
            captions = []
            captions.append(item["Caption_1"])
            captions.append(item["Caption_2"])
            captions = [c for c in captions if isinstance(c, str) and c.strip()]
            aide_unique[img_id] = captions

    intersection_ids = set(k for k in aide_unique.keys() if k in flickr_polish_images)
    flickr_candidates = [item for item in data_Flickr if item[0].__str__() not in intersection_ids]

    val_set_flickr = flickr_candidates[:1000]
    test_set_flickr = flickr_candidates[1000:2000]
    train_set_flickr = flickr_candidates[2000:]

    with open("flickr30kPolish_captions_train.json", "w", encoding="utf-8") as dest:
        dest.write("[\n")
        for item in train_set_flickr:
            captions = []
            img_id = item[0]
            captions = item[1]
            json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
            dest.write(",\n")
        dest.seek(dest.tell() - 2, os.SEEK_SET)
        dest.write("]\n")
        print("Training set created.")

    with open("flickr30kPolish_captions_val.json", "w", encoding="utf-8") as dest:
        dest.write("[\n")
        for item in val_set_flickr:
            captions = []
            img_id = item[0]
            captions = item[1]
            json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
            dest.write(",\n")
        dest.seek(dest.tell() - 2, os.SEEK_SET)
        dest.write("]\n")
        print("Validation set created.")

    with open("flickr30kPolish_captions_test_std.json", "w", encoding="utf-8") as dest:
        dest.write("[\n")
        for item in test_set_flickr:
            captions = []
            img_id = item[0]
            captions = item[1]
            json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
            dest.write(",\n")
        dest.seek(dest.tell() - 2, os.SEEK_SET)
        dest.write("]\n")
        print("Test-STD set created.")

    with open("flickr30kPolish_captions_test_hq.json", "w", encoding="utf-8") as dest:
        dest.write("[\n")
        for img_id, captions in aide_unique.items():
            json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
            dest.write(",\n")
        dest.seek(dest.tell() - 2, os.SEEK_SET)
        dest.write("]\n")
        print("Test-HQ set created.")

print("AIDe_total:", len(aide_unique))
print("AIDe_intersection_with_Flickr:", len(intersection_ids))
print("Test-HQ set created.")