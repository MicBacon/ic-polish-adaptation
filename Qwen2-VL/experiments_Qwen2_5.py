import json
import argparse
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

from transformers import (AutoProcessor, AutoConfig,
                          TrainingArguments, Trainer)
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Utils from Qwen-VL (repo helper). If unavailable, we implement a fallback.
try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
except Exception:
    qwen_process_vision_info = None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            data.append(json.loads(ln))
    return data


class CaptionJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples = read_jsonl(jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        image_path = ex["image"]
        conv = ex["conversations"]
        assert isinstance(conv, list) and len(conv) >= 2, "conversations must be [human, gpt]"

        user_text = conv[0]["value"]
        asst_text = conv[1]["value"]

        # Build messages in the repo's "chat" format used by AutoProcessor.apply_chat_template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text.replace("<image>", "").strip()},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": asst_text}
                ],
            },
        ]
        return {
            "messages": messages,
            "image_path": image_path,
            "assistant_text": asst_text  # for masking length reference if needed
        }


@dataclass
class DataCollatorQwenVL:
    """
    Collate that:
      1) Builds chat template with both user+assistant (single sequence),
      2) Tokenizes with images through AutoProcessor,
      3) Creates labels identical to input_ids, but masks (=-100) all user tokens.
    """
    processor: AutoProcessor
    device: torch.device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build texts and gather vision inputs
        texts = []
        all_messages = []
        images = []
        videos = []

        # We use Qwen's multi-modal chat template
        for item in batch:
            messages = item["messages"]

            # Qwen helper (if available) to separate vision inputs
            if qwen_process_vision_info is not None:
                image_inputs, video_inputs = qwen_process_vision_info(messages)
            else:
                # Fallback: extract image paths manually; no videos in our use case
                image_inputs = []
                for seg in messages[0]["content"]:
                    if seg.get("type") == "image":
                        image_inputs.append(seg["image"])
                video_inputs = None

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # we include assistant in the training text
            )

            texts.append(text)
            all_messages.append(messages)
            images.append(image_inputs)
            videos.append(video_inputs)

        # Process batch through processor (tokenize + vision)
        # Note: processor handles reading image paths when strings are passed in "images".
        proc_out = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )

        input_ids = proc_out["input_ids"]
        attention_mask = proc_out["attention_mask"]

        # Labels = input_ids but mask user tokens (-100)
        labels = input_ids.clone()

        # To mask user tokens, we rebuild a "prompt-only" version per item and measure its token length.
        # Strategy:
        #   - Take each "messages" and drop the assistant turn -> user-only chat.
        #   - Tokenize that prompt-only string; its length = prefix len to mask.
        #   - Mask labels[:, :prefix_len] = -100 (per row, careful with padding differences).
        for i, messages in enumerate(all_messages):
            prompt_only = [messages[0]]  # keep only the user turn
            prompt_text = self.processor.apply_chat_template(
                prompt_only,
                tokenize=False,
                add_generation_prompt=True  # generation prompt stops before assistant
            )
            # Tokenize prompt-only to get its tokenized length for this sample
            pt = self.processor(text=[prompt_text], padding=False, return_tensors="pt")
            prompt_len = pt["input_ids"].shape[1]
            # Clamp to the sequence length to be safe
            prompt_len = min(prompt_len, labels.shape[1])
            # Mask the user tokens in labels
            labels[i, :prompt_len] = -100

        batch_out = {
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        # Vision tensors (pixel_values, etc.) are already inside proc_out and on CPU;
        # move all remaining processor outputs to device:
        for k, v in proc_out.items():
            if k in batch_out:
                continue
            if isinstance(v, torch.Tensor):
                batch_out[k] = v.to(self.device)
        return batch_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True,
                    help="e.g., Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)  # smoke test default
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--no_flash_attn", action="store_true",
                    help="Disable FlashAttention-2 even if available.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model & processor
    model_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
    )
    if not args.no_flash_attn:
        # If flash-attn2 is installed, enable it
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[INFO] Loading model: {args.model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, **model_kwargs
    )

    processor = AutoProcessor.from_pretrained(args.model_name)

    # LoRA (no 4-bit here; H100 has plenty of VRAM)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Prepare and wrap
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Datasets
    train_ds = CaptionJsonlDataset(args.train_file)
    val_ds = CaptionJsonlDataset(args.val_file)

    data_collator = DataCollatorQwenVL(processor=processor, device=device)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # use BF16 on H100
        fp16=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # IMPORTANT for multimodal batches
        report_to="none",
        max_grad_norm=1.0,
    )

    # prepare metrics
    bleu = evaluate.load("bleu")
    cider = evaluate.load("cider")
    rouge = evaluate.load("rouge") #rouge_raw
    meteor = evaluate.load("meteor")
    spice = evaluate.load("spice")
    bertscore = evaluate.load("bertscore")
    clipscore = evaluate.load("clip_score")

    # Simple metrics: we log eval loss; (optional) can add CIDEr later.
    def compute_metrics(_eval_pred):
        processor = AutoProcessor.from_pretrained("your-model-checkpoint")
        predictions, labels = _eval_pred
        # Decode predictions and labels
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Prepare references for metrics (e.g., CIDEr and SPICE may require list of lists)
        references = [[label] for label in decoded_labels]  # For metrics expecting list of references per prediction
        
        # Compute metrics
        bleu_results = bleu.compute(predictions=decoded_preds, references=references)
        meteor_results = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        cider_results = cider.compute(predictions=decoded_preds, references=references)
        spice_results = spice.compute(predictions=decoded_preds, references=references)
        bertscore_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels)
        clipscore_results = clipscore.compute(predictions=decoded_preds, references=decoded_labels, model_type="ViT-L-14/openai")

        return {
            "bleu": bleu_results["bleu"],
            "meteor": meteor_results["meteor"],
            "rougeL": rouge_results["rougeL"],
            "cider": cider_results["cider"],
            "spice": spice_results["spice"],
            "bertscore": bertscore_results["bertscore"],
            "clipscore": clipscore_results["clipscore"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training finished.")

    # Save final adapter (LoRA) only
    trainer.model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"[DONE] Saved LoRA adapter & processor to: {args.out_dir}")


if __name__ == "__main__":
    main()