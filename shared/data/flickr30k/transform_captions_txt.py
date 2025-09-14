import json

with open("captions.txt", "r") as f:
    f.readline() # skip headers
    
    with open("flickr30k_captions_all.json", "w") as out:
        out.write("[\n")

        captions = []
        parts = f.readline().strip().split(",", 1)
        latest_img = parts[0]

        captions.append(parts[1].strip().strip('"'))

        for line in f:
            parts = line.strip().split(",", 1)

            if latest_img != parts[0]:
                json.dump({"image_id": latest_img, "captions": captions}, out, ensure_ascii=False)
                out.write(",\n")
                captions = []
                latest_img = parts[0]

            captions.append(parts[1].strip('"'))
        
        json.dump({"image_id": latest_img, "captions": captions}, out, ensure_ascii=False)

        out.write("]\n")