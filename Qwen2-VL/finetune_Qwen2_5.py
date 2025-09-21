import json
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
except Exception:
    qwen_process_vision_info = None

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE = ""
VAL_FILE = ""
OUT_DIR = "out"
EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.05
EVAL_STEPS = 500
SAVE_STEPS = 500
USE_FLASH_ATTN = True

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            data.append(json.loads(ln))
    return data

class CaptionJsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = read_jsonl(jsonl_path)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        ex = self.samples[idx]
        image_path = ex["image"]
        conv = ex["conversations"]
        user_text = conv[0]["value"]
        asst_text = conv[1]["value"]
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text.replace("<image>", "").strip()},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": asst_text}]},
        ]
        return {"messages": messages, "image_path": image_path, "assistant_text": asst_text}

class DataCollatorQwenVL:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        texts, all_messages, images, videos = [], [], [], []
        for item in batch:
            messages = item["messages"]
            if qwen_process_vision_info is not None:
                image_inputs, video_inputs = qwen_process_vision_info(messages)
            else:
                image_inputs = []
                for seg in messages[0]["content"]:
                    if seg.get("type") == "image":
                        image_inputs.append(seg["image"])
                video_inputs = None
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            all_messages.append(messages)
            images.append(image_inputs)
            videos.append(video_inputs)
        proc_out = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids = proc_out["input_ids"]
        attention_mask = proc_out["attention_mask"]
        labels = input_ids.clone()
        for i, messages in enumerate(all_messages):
            prompt_only = [messages[0]]
            prompt_text = self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
            pt = self.processor(text=[prompt_text], padding=False, return_tensors="pt")
            prompt_len = min(pt["input_ids"].shape[1], labels.shape[1])
            labels[i, :prompt_len] = -100
        batch_out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        for k, v in proc_out.items():
            if k in batch_out:
                continue
            if isinstance(v, torch.Tensor):
                batch_out[k] = v
        return batch_out

def main():
    if not TRAIN_FILE or not VAL_FILE:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    train_ds = CaptionJsonlDataset(TRAIN_FILE)
    val_ds = CaptionJsonlDataset(VAL_FILE)
    data_collator = DataCollatorQwenVL(processor=processor)
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
        prediction_loss_only=True,
        eval_accumulation_steps=2,
        include_inputs_for_metrics=False,
        disable_tqdm=True,
        log_level="error",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    trainer.model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(OUT_DIR)

if __name__ == "__main__":
    main()