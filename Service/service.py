import gradio as gr
import requests
from PIL import Image
import io

def produce_caption(image, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "mPLUG":
        return "mPLUG not attached"
    
    elif model_name == "Qwen2.5 (baseline)":
        return f"Qwen2.5 (baseline)"
    else:
        return f"Caption for {model_name} model"
    
demo = gr.Interface(
    fn=produce_caption,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Dropdown(choices=["mPLUG (Flickr30k only)", "mPLUG Full", "Qwen2.5-VL-7B (baseline) EN->PL", "Qwen2.5-VL-7B (baseline)", "Qwen2.5-VL-7B (finetuned)", "Qwen2.5-VL-7B (extended)"], label="Model")
    ],
    outputs=gr.Textbox(label="Caption")
)

demo.launch(server_name="0.0.0.0", server_port=7861)