import gradio as gr
import requests
from PIL import Image
import io

def produce_caption(image, model_name):
    match model_name:
        case "mPLUG (Flickr30k only)":
            
        case "mPLUG (Full)":

        case "Qwen2.5-VL-7B (baseline) EN->PL":

        case "Qwen2.5-VL-7B (baseline)": 

        case "Qwen2.5-VL-7B (finetuned)":

        case "Qwen2.5-VL-7B (extended)":
        case _:



    
demo = gr.Interface(
    fn=produce_caption,
    inputs=[
        gr.Image(label="Input Image", type="pil"), 
        gr.Dropdown(choices=["mPLUG (Flickr30k only)", "mPLUG (Full)", "Qwen2.5-VL-7B (baseline) EN->PL", "Qwen2.5-VL-7B (baseline)", "Qwen2.5-VL-7B (extended)", "Qwen2.5-VL-7B (finetuned)"], label="Model variant")
    ],
    outputs=gr.Textbox(label="Caption")
)

demo.launch(server_name="0.0.0.0", server_port=7861, auth=('tester', 'test'))