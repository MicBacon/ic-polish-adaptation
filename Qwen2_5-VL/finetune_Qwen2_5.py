import os, sys
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, TrainerCallback
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.path.dirname('/workspace/'))

from shared.MetricComputer import MetricComputer
import wandb

try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
except Exception:
    qwen_process_vision_info = None

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE = "../shared/data/flickr30k/flickr30kPolish_captions_train.json"
VAL_FILE = "../shared/data/flickr30k/flickr30kPolish_captions_val.json"
IMAGE_ROOT = "/workspace/shared/data/flickr30k" 
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
MAX_NEW_TOKENS = 32
NUM_BEAMS = 3
TEMPERATURE = 0.0

VAL_EVAL_N = 50
WANDB_LOG_SAMPLES = 4

os.environ["WANDB_PROJECT"] = "magisterka"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

mc = MetricComputer()

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def image_id_from_path(p):
    b = os.path.basename(p)
    return os.path.splitext(b)[0]

class ICMetricsCallback(TrainerCallback):
    def __init__(self, processor, val_ds, refs_map, n_samples=0, log_samples=0):
        self.p = processor
        self.ds = val_ds
        self.refs = refs_map
        self.n = n_samples
        self.log_samples = log_samples
        self._trainer = None
        self.mc = mc

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        if self._trainer is None or self._trainer.model is None:
            return control
        model = self._trainer.model.eval()
        ds = [self.ds[i] for i in range(min(self.n, len(self.ds)))] if (self.n and self.n > 0) else self.ds
        preds, refs, imgs = do_eval_generate(model, self.p, ds, self.refs)
        m = self.mc.compute_metrics(preds, refs, imgs)
        logs = {f"eval_{k}": float(v) for k, v in m.items() if isinstance(v, (int, float))}
        logs["eval_callback_ping"] = 1.0
        self._trainer.log(logs)
        try:
            wandb.log(logs, step=state.global_step)
            if self.log_samples and self.log_samples > 0:
                table = wandb.Table(columns=["image","pred","ref"])
                for i in range(min(self.log_samples, len(preds))):
                    table.add_data(wandb.Image(imgs[i]), preds[i], refs[i][0] if refs[i] else "")
                wandb.log({"eval_samples": table}, step=state.global_step)
        except Exception:
            pass
        return control

class EarlyStopByMetric(TrainerCallback):
    def __init__(self, metric_name, greater_is_better=True, patience=3, save_best_dir=None):
        self.m = metric_name
        self.g = greater_is_better
        self.p = patience
        self.best = None
        self.wait = 0
        self.save_best_dir = save_best_dir
        self._trainer = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.m not in logs:
            return control
        v = logs[self.m]
        improved = (self.best is None) or (v > self.best if self.g else v < self.best)
        if improved:
            self.best = v
            self.wait = 0
            if self.save_best_dir and self._trainer is not None:
                os.makedirs(self.save_best_dir, exist_ok=True)
                self._trainer.save_model(self.save_best_dir)
        else:
            self.wait += 1
            if self.wait >= self.p:
                control.should_training_stop = True
        return control

def _find_image_path(image_id, image_root):
    exts = ["jpg","jpeg","png"]
    s = str(image_id)
    cands = []
    for ext in exts:
        if image_root:
            cands.append(os.path.join(image_root, f"{s}.{ext}"))
            cands.append(os.path.join(image_root, "Images", f"{s}.{ext}"))
    for p in cands:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return os.path.abspath(os.path.join(image_root, "Images", f"{s}.jpg"))

class CaptionJsonlDataset(Dataset):
    def __init__(self, json_path, image_root=""):
        raw = read_json(json_path)
        self.samples = []
        self.image_root = image_root
        for rec in raw:
            img_id = rec.get("image_id")
            caps = rec.get("captions") or []
            ip = _find_image_path(img_id, self.image_root)
            for c in caps:
                if isinstance(c, str) and c.strip():
                    self.samples.append({"image_path": ip, "assistant_text": c.strip(), "image_id": str(img_id)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ex["assistant_text"]}]},
        ]
        return {"messages": messages, "image_path": ex["image_path"], "assistant_text": ex["assistant_text"], "image_id": ex["image_id"]}

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
        mm_mask = proc_out.get("mm_token_mask", None)
        if mm_mask is not None:
            labels[mm_mask.bool()] = -100
        else:
            tok = self.processor.tokenizer
            image_id = getattr(tok, "image_token_id", None)
            video_id = getattr(tok, "video_token_id", None)
            if image_id is not None:
                labels[input_ids == image_id] = -100
            if video_id is not None:
                labels[input_ids == video_id] = -100
        for i, messages in enumerate(all_messages):
            prompt_only = [messages[0], messages[1]]
            prompt_text = self.processor.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            )
            if qwen_process_vision_info is not None:
                prompt_images, _ = qwen_process_vision_info(prompt_only)
            else:
                prompt_images = []
                for seg in messages[1]["content"]:
                    if seg.get("type") == "image":
                        prompt_images.append(seg["image"])
            pt = self.processor(text=[prompt_text], images=[prompt_images], padding=False, return_tensors="pt")
            prompt_len = min(pt["input_ids"].shape[1], labels.shape[1])
            labels[i, :prompt_len] = -100
        batch_out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        for k, v in proc_out.items():
            if k in batch_out:
                continue
            if isinstance(v, torch.Tensor):
                batch_out[k] = v
        return batch_out

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
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"device_map": "auto"}
    if USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb_config, **model_kwargs)
    model = prepare_model_for_kbit_training(model)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    refs_map = {}
    if VAL_FILE:
        raw = read_json(VAL_FILE)
        if isinstance(raw, dict):
            refs_map = raw
        elif isinstance(raw, list):
            for rec in raw:
                k = str(rec.get("image_id", ""))
                rs = rec.get("captions") or rec.get("references") or []
                if k:
                    refs_map[k] = rs
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
    train_ds = CaptionJsonlDataset(TRAIN_FILE, image_root=IMAGE_ROOT)
    val_ds   = CaptionJsonlDataset(VAL_FILE, image_root=IMAGE_ROOT)
    data_collator = DataCollatorQwenVL(processor=processor)
    ic_cb = ICMetricsCallback(processor, val_ds, refs_map, n_samples=VAL_EVAL_N, log_samples=WANDB_LOG_SAMPLES)
    es_cb = EarlyStopByMetric("eval_BERTScore_F1", greater_is_better=True,
                          patience=8, save_best_dir=os.path.join(OUT_DIR, "best_by_BERTScore_2"))
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
        max_grad_norm=1.0,
        prediction_loss_only=True,
        eval_accumulation_steps=2,
        include_inputs_for_metrics=False,
        disable_tqdm=True,
        log_level="error",
        report_to="wandb",
        run_name="qwen2.5-vl-finetune_BSEarlyStopping_patience_8_new_metrics_quantization",)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[ic_cb, es_cb]
    )
    ic_cb.set_trainer(trainer)
    es_cb.set_trainer(trainer)
    trainer.evaluate()
    trainer.train()
    trainer.model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(OUT_DIR)
    model.eval()
    preds, refs, img_paths = do_eval_generate(model, processor, val_ds, refs_map)
    metrics = mc.compute_metrics(preds, refs, img_paths)
    mpath = os.path.join(OUT_DIR, "val_metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(mpath)
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()