import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
import wandb

try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
except Exception:
    qwen_process_vision_info = None

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE = "../shared/data/flickr30k/flickr30kPolish_train.jsonl"
VAL_FILE = "../shared/data/flickr30k/flickr30kPolish_val.jsonl"
OUT_DIR = "out"
EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.05
EVAL_STEPS = 500
SAVE_STEPS = 500
USE_FLASH_ATTN = True
SYSTEM_PROMPT = "Jesteś ekspertem od opisu obrazów. Pisz po polsku, jasno i bez halucynacji."
USER_PROMPT = "Opisz ten obraz w 1 zdaniu. Uwzględnij obiekty, relacje i tło. Nie zgaduj."
MAX_NEW_TOKENS = 64
NUM_BEAMS = 3
TEMPERATURE = 0.0
EVAL_REFS_JSON = "../shared/data/flickr30k/flickr30kPolish_captions_val.json"        
COMPUTE_METRICS_PY = "../shared/compute_metrics.py"

os.environ["WANDB_PROJECT"] = "magisterka"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def image_id_from_path(p):
    b = os.path.basename(p)
    return os.path.splitext(b)[0]

class CaptionJsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = read_jsonl(jsonl_path)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        ex = self.samples[idx]
        image_path = ex["image"]
        conv = ex["conversations"]
        asst_text = conv[1]["value"]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": asst_text}]},
        ]
        return {"messages": messages, "image_path": image_path, "assistant_text": asst_text, "image_id": image_id_from_path(image_path)}

class DataCollatorQwenVL:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        texts, all_messages, images = [], [], []
        for item in batch:
            messages = item["messages"]
            if qwen_process_vision_info is not None:
                image_inputs, _ = qwen_process_vision_info(messages)
            else:
                image_inputs = []
                for seg in messages[1]["content"]:
                    if seg.get("type") == "image":
                        image_inputs.append(seg["image"])
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            all_messages.append(messages)
            images.append(image_inputs)
        proc_out = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids = proc_out["input_ids"]
        attention_mask = proc_out["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for i, messages in enumerate(all_messages):
            prompt_only = [messages[0], messages[1]]
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

def load_compute_metrics(module_path):
    if module_path:
        path = os.path.abspath(module_path)
        if os.path.isfile(path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("compute_metrics", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            if hasattr(mod, "compute_metrics"):
                return mod.compute_metrics
    def fallback_metrics(preds, refs, image_paths=None):
        out = {}
        try:
            import sacrebleu
            max_k = max(len(r) for r in refs) if refs else 0
            ref_sets = []
            for k in range(max_k):
                ref_sets.append([(r[k] if k < len(r) else r[-1]) for r in refs])
            bleu = sacrebleu.corpus_bleu(preds, ref_sets, tokenize="intl")
            out["SacreBLEU"] = float(bleu.score)
        except Exception:
            pass
        try:
            from bert_score import score as bert_score
            first_refs = [r[0] if len(r) > 0 else "" for r in refs]
            _, _, F1 = bert_score(preds, first_refs, lang="pl", rescale_with_baseline=True)
            out["BERTScore_F1"] = float(F1.mean().item())
        except Exception:
            pass
        out["Len_pred_tokens_avg"] = sum(len(p.split()) for p in preds) / max(1, len(preds))
        return out
    return fallback_metrics

def do_eval_generate(model, processor, ds, refs_map):
    device = next(model.parameters()).device
    preds, refs, img_paths = [], [], []
    for ex in ds:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if qwen_process_vision_info is not None:
            images, _ = qwen_process_vision_info(messages)
        else:
            images = [[ex["image_path"]]]
        inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0.0),
                temperature=(TEMPERATURE if TEMPERATURE > 0.0 else None),
                num_beams=(NUM_BEAMS if NUM_BEAMS and NUM_BEAMS > 1 and TEMPERATURE == 0.0 else 1),
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, in_len:]
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        preds.append(pred)
        img_paths.append(ex["image_path"])
        k = ex["image_id"]
        rs = refs_map.get(k) or [ex["assistant_text"]]
        rs = [r.strip() for r in rs if isinstance(r, str)] or [""]
        refs.append(rs)
    return preds, refs, img_paths

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
        eval_strategy="steps",
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
        report_to="wandb",
        run_name="qwen2.5-vl-finetune",)
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
    refs_map = {}
    if EVAL_REFS_JSON:
        raw = read_json(EVAL_REFS_JSON)
        if isinstance(raw, dict):
            refs_map = raw
        elif isinstance(raw, list):
            for rec in raw:
                k = str(rec.get("image_id", ""))
                rs = rec.get("captions") or rec.get("references") or []
                if k:
                    refs_map[k] = rs
    model.eval()
    preds, refs, img_paths = do_eval_generate(model, processor, val_ds, refs_map)
    metrics_fn = load_compute_metrics(COMPUTE_METRICS_PY if COMPUTE_METRICS_PY else "")
    metrics = metrics_fn(preds, refs, img_paths)
    mpath = os.path.join(OUT_DIR, "val_metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(mpath)
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()