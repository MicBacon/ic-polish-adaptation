import gradio as gr
import requests
from PIL import Image
import io

MPLUG_URL = "http://gradio-mplug-ctn:7863/generate_caption"
QWEN_URL = "http://gradio-qwen-ctn:7862/generate_caption"

def produce_caption(image_pil, variant_name):
    if not image_pil:
        return '[ERROR] You must upload an image.'
    if not variant_name:
        return '[ERROR] You must specify a variant in dropdownlist.'
    
    if 'mplug' in variant_name:
        url = MPLUG_URL
    elif 'qwen' in variant_name:
        url = QWEN_URL    
    else:
        return '[ERROR] Variant not supported. Contact author.'

    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    try:
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        data  = {'model': variant_name}
        response = requests.post(url, files=files, data=data, timeout=120)
        if response.status_code == 200:
            return response.json().get("caption", "[ERROR] Error getting json response.")
        else:
            return f"[ERROR] Not OK ({response.status_code}): {response.text}"
            
    except requests.exceptions.ConnectionError:
        return f"[ERROR] Can't connect to {variant_name} server."
    except Exception as e:
        return f"[ERROR] {e}"
    
CHOICES = [
    ("mPLUG (Flickr30k only)", "mplug-flickr-only"),
    ("mPLUG (Full)", "mplug-full"),
    ("Qwen2.5-VL-7B EN->PL", "qwen-en2pl"),
    ("Qwen2.5-VL-7B (baseline)", "qwen-baseline"),
    ("Qwen2.5-VL-7B (extended)", "qwen-ext"),
    ("Qwen2.5-VL-7B (finetuned)", "qwen-ft"),
]

app = gr.Interface(
    fn=produce_caption,
    inputs=[
        gr.Image(label="Input Image", type="pil"), 
        gr.Dropdown(choices=CHOICES, label="Model variant", value=CHOICES[0][1],
                    info="Choose one of the model variants from paper then click 'Submit' to produce caption.")
    ],
    outputs=gr.Textbox(label="Caption", lines=10)
)

app.launch(server_name="0.0.0.0", server_port=7861, auth=('tester', 'test'))
#app.launch(server_port=7861, share=True, auth=('tester', 'test'))