import json

with open('flickr30k_captions_all.json', 'r') as f:
    data = json.load(f)

    # Split the data into training, validation, and test sets
    train_data = data[:25000]
    val_data = data[25000:28000]
    test_data = data[28000:]

    with open('flickr30k_captions_train.json', 'w') as f:
        json.dump(train_data, f)
    with open('flickr30k_captions_val.json', 'w') as f:
        json.dump(val_data, f)
    with open('flickr30k_captions_test.json', 'w') as f:
        json.dump(test_data, f)