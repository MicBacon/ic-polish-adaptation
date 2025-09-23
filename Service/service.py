import gradio as gr
import sys

# awkward solution because mPLUG is visible but modules inside are not TODO: fix it properly
#sys.path.insert(0, '/Users/michalboczon/dev/Magisterka/ic-polish-adaptation/mPLUG/')

#from mPLUG.models.model_caption_mplug import MPLUG
#from mPLUG.models.tokenization_bert import BertTokenizer
#from mPLUG.models.vit import resize_pos_embed
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import ruamel.yaml as yaml

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = "Jesteś ekspertem od opisu obrazów. Pisz po polsku, jasno i bez halucynacji."
USER_PROMPT = "Opisz ten obraz w 1 zdaniu. Uwzględnij obiekty, relacje i tło. Nie zgaduj."

def produce_caption(image, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "mPLUG":
        # object_labels = []
        # config = yaml.load(open('../mPLUG/configs/caption_mplug_large.yaml', 'r'), Loader=yaml.Loader)
        # max_input_length = 25
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # config['bert_config'] = '../mPLUG/' + config['bert_config']
        # config['text_encoder'] = 'bert-base-uncased'
        # config['text_decoder'] = 'bert-base-uncased'
        # config["min_length"] = 1
        # config["max_length"] = 10
        # config["add_object"] = False
        # config["beam_size"] = 5
            
        # model = MPLUG(tokenizer=tokenizer, config=config).to(device)
        # model.eval()
        # # TODO: change for trained checkpoint
        # checkpoint = torch.load('../mPLUG/mPLUG_large_v2.pth', map_location='cpu')
        # try:
        #     state_dict = checkpoint['model']
        # except:
        #     state_dict = checkpoint['module']

        # if config["clip_name"] == "ViT-B-16":
        #     num_patches = int(config["image_res"] * config["image_res"]/(16*16))
        # elif config["clip_name"] == "ViT-L-14":
        #     num_patches = int(config["image_res"] * config["image_res"]/(14*14))
        # pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        # pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
        #                                            pos_embed.unsqueeze(0))
        # state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        # model.load_state_dict(state_dict, strict=False)

        # normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # test_transform = transforms.Compose([
        #     transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC), 
        #     transforms.ToTensor(),
        #     normalize,
        # ])  

        # image=test_transform(image)
        # image = image.to(device,non_blocking=True)             
        # question_input = [config['bos']+" "+each for each in object_labels]
        # question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=max_input_length, return_tensors="pt").to(device)
        # topk_ids, topk_probs = model(image, question_input, train=False)
        # result = []
        # for topk_id, topk_prob in zip(topk_ids, topk_probs):
        #     ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
        #     result.append(ans)   
            
        # return result[0]
        return "mPLUG not attached"
    
    elif model_name == "Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    
    else:
        return f"Caption for {model_name} model"

demo = gr.Interface(
    fn=produce_caption,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Dropdown(choices=["mPLUG", "Qwen2.5-VL-7B-Instruct", "InternVL2_5-8B"], label="Model")
    ],
    outputs=gr.Textbox(label="Caption") #TODO add metric scores
)

demo.launch()