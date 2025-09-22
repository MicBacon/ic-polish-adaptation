import gradio as gr
import sys

# awkward solution because mPLUG is visible but modules inside are not TODO: fix it properly
sys.path.insert(0, '/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/mPLUG/')

from mPLUG.models.model_caption_mplug import MPLUG
from mPLUG.models.tokenization_bert import BertTokenizer
from mPLUG.models.vit import resize_pos_embed
import torch
import torch.nn as nn
import ruamel.yaml as yaml

def produce_caption(img, model_name):
    if model_name == "mPLUG":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        object_labels = []
        config = yaml.load(open('../mPLUG/configs/caption_mplug_large.yaml', 'r'), Loader=yaml.Loader)
        max_input_length = 25
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config['bert_config'] = '../mPLUG/' + config['bert_config']
        config['text_encoder'] = 'bert-base-uncased'
        config['text_decoder'] = 'bert-base-uncased'
        config["min_length"] = 1
        config["max_length"] = 10
        config["add_object"] = False
        config["beam_size"] = 5
            
        model = MPLUG(tokenizer=tokenizer, config=config).to(device)
        model.eval()
        # TODO: change for trained checkpoint
        checkpoint = torch.load('../mPLUG/mPLUG_large_v2.pth', map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"]/(16*16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"]/(14*14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        model.load_state_dict(state_dict, strict=False)

        image = img.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']+" "+each for each in object_labels]
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=max_input_length, return_tensors="pt").to(device)
        topk_ids, topk_probs = model(image, question_input, caption, train=False)
        result = []
        for topk_id, topk_prob in zip(topk_ids, topk_probs):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append(ans)   
        return result[0]
    elif model_name == "Qwen2.5-VL-7B-Instruct":
        return f"Caption for Qwen2.5-VL-7B-Instruct model"
    else:
        return f"Caption for {model_name} model"

demo = gr.Interface(
    fn=produce_caption,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Dropdown(choices=["mPLUG", "Qwen2.5-VL-7B-Instruct", "InternVL2_5-8B"], label="Model")
    ],
    outputs=gr.Textbox(label="Caption")
)
demo.launch()