import gradio as gr
import requests
from PIL import Image
import io

MPLUG_URL = "http://gradio-mplug-ctn:7863/generate_caption"
QWEN_URL = "http://gradio-qwen-ctn:7862/generate_caption"

CHOICES = [
    ("mPLUG (Flickr30k only)", "mplug-flickr-only"),
    ("mPLUG (Full)", "mplug-full"),
    ("Qwen2.5-VL-7B EN->PL", "qwen-en2pl"),
    ("Qwen2.5-VL-7B (baseline)", "qwen-baseline"),
    ("Qwen2.5-VL-7B (extended)", "qwen-ext"),
    ("Qwen2.5-VL-7B (finetuned)", "qwen-ft"),
]
DEFAULT_VARIANT = CHOICES[0][1]

def produce_caption(image_pil, variant_name):
    if not image_pil:
        return 'You must upload an image.'
    if 'mplug' in variant_name:
        url = MPLUG_URL
    elif 'qwen' in variant_name:
        url = QWEN_URL
    else:
        return 'You must specify a variant in dropdownlist.'

    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    try:
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        data  = {'model': variant_name}
        response = requests.post(url, files=files, data=data, timeout=120)
        if response.status_code == 200:
            return response.json().get("caption", "Error getting json response.")
        else:
            return f"Not OK ({response.status_code}): {response.text}"
    except requests.exceptions.ConnectionError:
        return f"Can't connect to {variant_name} server."
    except Exception as e:
        return f"Unexpected error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## Image Captioning")
    with gr.Row():
        image = gr.Image(label="Input Image", type="pil")
        variant = gr.Dropdown(
            choices=CHOICES,
            value=DEFAULT_VARIANT,
            label="Model variant",
            info="Choose one of the model variants from the paper then click 'Submit' to produce a caption."
        )
    caption = gr.Textbox(label="Caption", lines=10)
    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear")

    submit.click(produce_caption, [image, variant], caption)

    def reset_fn():
        return None, gr.update(value=DEFAULT_VARIANT), ""
    clear.click(reset_fn, outputs=[image, variant, caption])

demo.launch()