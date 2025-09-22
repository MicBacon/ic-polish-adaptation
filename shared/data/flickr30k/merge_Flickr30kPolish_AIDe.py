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
    # split AIDe annotations randomly with fixed seed as training 100, validation 100 and training 800
    print("Building training, validation and test sets...")
    random.seed(42)

    random.shuffle(data_AIDe)

    # 1000 images annotated
    train_set_AIDe = data_AIDe[:800]
    val_set_AIDe = data_AIDe[800:900]
    test_set_AIDe = data_AIDe[900:]

    random.shuffle(data_Flickr)

    # 31534 images annotated 
    #train_set_flickr = data_Flickr[:25227] # ~80 %
    #val_set_flickr = data_Flickr[25227:29957] # ~15%
    #test_set_flickr = data_Flickr[29957:] # ~5%

    # stick with just train set for flickr, so 800 of images will have 7 captions in training set
    train_set_flickr = data_Flickr # 100%

    with open("flickr30kPolish_captions_train.json", "w") as dest:
        dest.write("[\n")
        
        # first process merging of 800 annotations
        for item in train_set_AIDe:
            captions = []
            # get image id
            img_id = item["Picture_orig_name"].split("_")[0] # remove suffixes after _

            captions.append(item["Caption_1"])
            captions.append(item["Caption_2"])

            # find image id in flickr set
            for flickr_item in train_set_flickr:
                if flickr_item[0].__str__() == img_id:
                    for caption in flickr_item[1]:
                        captions.append(caption)
                    
                    # delete matched item from flickr
                    train_set_flickr.remove(flickr_item)

            json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
            dest.write(",\n")
        
        # add rest flickr items to train set
        for item in train_set_flickr:
            captions = []
            img_id = item[0]
            captions = item[1]

            if img_id.__str__() not in [item["Picture_orig_name"].split("_")[0] for item in val_set_AIDe] and img_id.__str__() not in [item["Picture_orig_name"].split("_")[0] for item in test_set_AIDe]:
                json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
                dest.write(",\n")

        #delete comma from last item
        dest.seek(dest.tell() - 2, os.SEEK_SET)
        dest.write("]\n")

        print("Training set created.")

        with open("flickr30kPolish_captions_val.json", "w") as dest:
            dest.write("[\n")
            for item in val_set_AIDe:
                captions = []
                # get image id
                img_id = item["Picture_orig_name"].split("_")[0] # remove suffixes after _

                captions.append(item["Caption_1"])
                captions.append(item["Caption_2"])

                json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
                dest.write(",\n")

            dest.seek(dest.tell() - 2, os.SEEK_SET)
            dest.write("]\n")

        print("Validation set created.")

        with open("flickr30kPolish_captions_test.json", "w") as dest:
            dest.write("[\n")
            for item in test_set_AIDe:
                captions = []
                # get image id
                img_id = item["Picture_orig_name"].split("_")[0] # remove suffixes after _

                captions.append(item["Caption_1"])
                captions.append(item["Caption_2"])

                json.dump({"image_id": img_id, "captions": captions}, dest, ensure_ascii=False)
                dest.write(",\n")

            dest.seek(dest.tell() - 2, os.SEEK_SET)
            dest.write("]\n")
        
        print("Test set created.")
