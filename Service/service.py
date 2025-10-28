import gradio as gr

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
        gr.Dropdown(choices=["mPLUG", "mPLUGFull", "Qwen2.5 (baseline)", "Qwen2.5 (finetuned)", "Qwen2.5 (extended)"], label="Model")
    ],
    outputs=gr.Textbox(label="Caption")
)

demo.launch()