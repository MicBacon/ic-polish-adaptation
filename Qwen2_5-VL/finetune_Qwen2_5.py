import os, sys
import json
import torch
import random
from torch.utils.data import Dataset, Subset
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE = "../shared/data/flickr30k/flickr30kPolish_captions_train.json"
VAL_FILE = "../shared/data/flickr30k/flickr30kPolish_captions_val.json"
IMAGE_ROOT = "/workspace/shared/data/flickr30k" 
OUT_DIR = "out"
EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.05
EVAL_STEPS = 1500
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
    
class RandomSubsetEvalCallback(TrainerCallback):
    def __init__(self, base_eval_ds, refs_map, processor,
                 n_samples=50, every_steps=1500, gen_bs=16, fast_eval=True, seed=1337):
        self.base = base_eval_ds           # np. val_ds_loss (1 wpis/obraz)
        self.refs_map = refs_map
        self.p = processor
        self.n = n_samples
        self.every = every_steps
        self.gen_bs = gen_bs
        self.fast = fast_eval
        self.seed = seed
        self._trainer = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return control
        if self._trainer is None or self._trainer.model is None:
            return control

        rng = random.Random(self.seed + int(state.global_step))
        idxs = rng.sample(range(len(self.base)), min(self.n, len(self.base)))

        subset = Subset(self.base, idxs)
        loss_metrics = self._trainer.evaluate(eval_dataset=subset, metric_key_prefix="eval")  # nie blokuje treningu

        ds_list = [self.base[i] for i in idxs]
        num_beams = 1 if self.fast else (NUM_BEAMS or 1)
        max_new = 16 if self.fast else MAX_NEW_TOKENS

        preds, refs, imgs = do_eval_generate(
            self._trainer.model, self.p, ds_list, self.refs_map,
            num_beams=num_beams, max_new_tokens=max_new,
            temperature=0.0, gen_bs=self.gen_bs, use_cache=True
        )

        m = mc.compute_metrics_fast(preds, refs, imgs)

        logs = {**{f"eval_{k}": float(v) for k, v in m.items() if isinstance(v, (int, float))},
                **{k: float(v) for k, v in loss_metrics.items() if isinstance(v, (int, float))},
                "eval_N_samples": float(len(idxs))}
        self._trainer.log(logs)
        try:
            wandb.log(logs, step=state.global_step)
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

class CaptionDatasetOneCapPerImage(Dataset):
    def __init__(self, json_path, image_root=""):
        raw = read_json(json_path)
        self.samples = []
        self.image_root = image_root
        for rec in raw:
            img_id = rec.get("image_id")
            caps = rec.get("captions") or []
            if not caps: 
                continue
            ip = _find_image_path(img_id, self.image_root)
            c = caps[0].strip()
            if c:
                self.samples.append({"image_path": ip, "assistant_text": c, "image_id": str(img_id)})

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        ex = self.samples[idx]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text",  "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ex["assistant_text"]}]},
        ]
        return {
            "messages": messages,
            "image_path": ex["image_path"],
            "assistant_text": ex["assistant_text"],
            "image_id": ex["image_id"]
        }

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
        self.tok = processor.tokenizer

    def __call__(self, batch):
        texts, images, all_messages = [], [], []

        for item in batch:
            messages = item["messages"]

            if qwen_process_vision_info is not None:
                image_inputs, _ = qwen_process_vision_info(messages)
            else:
                image_inputs = [seg["image"] for seg in messages[1]["content"] if seg.get("type") == "image"]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            texts.append(text)
            images.append(image_inputs)
            all_messages.append(messages)

        proc_out = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids = proc_out["input_ids"]
        attention_mask = proc_out["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        mm_mask = proc_out.get("mm_token_mask", None)
        if mm_mask is not None:
            labels[mm_mask.bool()] = -100
        else:
            image_id = getattr(self.tok, "image_token_id", None)
            video_id = getattr(self.tok, "video_token_id", None)
            if image_id is not None:
                labels[input_ids == image_id] = -100
            if video_id is not None:
                labels[input_ids == video_id] = -100

        prompt_texts = []
        for m in all_messages:
            prompt_only = [m[0], m[1]]
            prompt_texts.append(self.processor.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            ))

        enc = self.tok(prompt_texts, add_special_tokens=False)
        prompt_lens = [len(x) for x in enc["input_ids"]]
        max_len = labels.shape[1]
        for i, L in enumerate(prompt_lens):
            labels[i, :min(L, max_len)] = -100

        batch_out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        for k, v in proc_out.items():
            if k not in batch_out and isinstance(v, torch.Tensor):
                batch_out[k] = v
        return batch_out


def do_eval_generate(
    model,
    processor,
    ds,
    refs_map,
    num_beams=1,                
    max_new_tokens=24,           
    temperature=0.0,
    gen_bs=4,                    
    use_cache=True               
):
    device = next(model.parameters()).device
    prev_cache = getattr(model.config, "use_cache", None)
    if use_cache:
        model.config.use_cache = True

    preds, refs, img_paths = [], [], []
    model.eval()

    def _extract_images(messages):
        if qwen_process_vision_info is not None:
            imgs, _ = qwen_process_vision_info(messages)
            return imgs
        return [seg["image"] for seg in messages[1]["content"] if seg.get("type") == "image"]

    for i in range(0, len(ds), gen_bs):
        chunk = ds[i:i+gen_bs]

        texts, batch_images = [], []
        for ex in chunk:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image_path"]},
                    {"type": "text",  "text": USER_PROMPT},
                ]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            texts.append(text)
            batch_images.append(_extract_images(messages))

        inputs = processor(text=texts, images=batch_images, padding=True, return_tensors="pt")
        input_lens = inputs["attention_mask"].sum(-1)

        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=(temperature if temperature > 0.0 else None),
                num_beams=num_beams,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=use_cache,
            )

        for j, ex in enumerate(chunk):
            start = input_lens[j].item()
            gen_ids = out_ids[j, start:]
            pred = processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
            preds.append(pred)
            img_paths.append(ex["image_path"])
            k = ex["image_id"]
            rs = refs_map.get(k) or [ex["assistant_text"]]
            rs = [r.strip() for r in rs if isinstance(r, str)] or [""]
            refs.append(rs)

    if prev_cache is not None:
        model.config.use_cache = prev_cache
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
    val_ds = CaptionDatasetOneCapPerImage(VAL_FILE, image_root=IMAGE_ROOT)
    rnd_cb = RandomSubsetEvalCallback(
        base_eval_ds=val_ds,
        refs_map=refs_map,
        processor=processor,
        n_samples=50,     
        every_steps=EVAL_STEPS,
        gen_bs=16, fast_eval=True, seed=1337
    )
    data_collator = DataCollatorQwenVL(processor=processor)
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
        save_strategy="steps",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        prediction_loss_only=True,
        eval_accumulation_steps=2,
        include_inputs_for_metrics=False,
        disable_tqdm=True,
        log_level="info",
        report_to="wandb",
        run_name="qwen2.5-vl-finetune_BSEarlyStopping_patience_8_new_metrics_quantization",)
    #model.gradient_checkpointing_enable()
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[rnd_cb, es_cb]
    )
    rnd_cb.set_trainer(trainer)
    es_cb.set_trainer(trainer)
    trainer.evaluate()
    trainer.train()
    trainer.model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(OUT_DIR)
    model.eval()
    preds, refs, img_paths = do_eval_generate(model, processor, val_ds, refs_map,
        num_beams=1, max_new_tokens=24, gen_bs=16, use_cache=True
    )
    metrics = mc.compute_metrics_fast(preds, refs, img_paths)
    mpath = os.path.join(OUT_DIR, "val_metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(mpath)
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()