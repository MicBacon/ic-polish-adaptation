import os, sys, re
import json, random
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, TrainerCallback
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from pathlib import Path
import wandb

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    from shared.MetricComputer import MetricComputer
except Exception:
    sys.path.append("../shared")
    from MetricComputer import MetricComputer

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
EPOCHS = 15
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 1e-5
WARMUP_RATIO = 0.1
SAVE_STEPS = 1000
USE_FLASH_ATTN = True
PEFT_ADAPTER_PATH = "out/best_by_CIDEr"
SYSTEM_PROMPT = "Jesteś ekspertem od opisu obrazów. Odpowiadasz wyłącznie w języku polskim. Napisz dokładnie jedno, pełne zdanie i zakończ je kropką. Nie zaczynaj drugiego zdania. Nie zgaduj."
USER_PROMPT = "Opisz ten obraz w jednym zdaniu: kluczowe obiekty, relacje i tło. Tylko po polsku, jedno zdanie, koniec po kropce."
MIN_NEW_TOKENS = 4
MAX_NEW_TOKENS = 128
NUM_BEAMS = 1
TEMPERATURE = 0.0

os.environ["WANDB_PROJECT"] = "magisterka"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

mc = MetricComputer()

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def image_id_from_path(p):
    b = os.path.basename(p)
    return os.path.splitext(b)[0]

def one_sentence(s: str) -> str:
    s = s.split("<|im_end|>")[0].splitlines()[0].strip()
    s = re.split(r'(?<=[\.\!\?])\s+', s)[0].strip()
    s = re.sub(r'^[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+', '', s).strip()
    return s

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

def _extract_images_from_messages(messages):
    if qwen_process_vision_info is not None:
        imgs, _ = qwen_process_vision_info(messages)
        return imgs
    out = []
    for seg in messages[1]["content"]:
        if seg.get("type") == "image":
            out.append(Image.open(seg["image"]).convert("RGB"))
    return out

def normalize_records(raw_list, image_root):
    out = []
    for rec in raw_list:
        caps = rec.get("captions") or rec.get("sentences") or rec.get("caption")
        if isinstance(caps, str):
            caps = [caps]
        caps = [c.strip() for c in (caps or []) if isinstance(c, str) and c.strip()]
        if not caps:
            continue
        img_id = rec.get("image_id") or rec.get("id") or rec.get("imageId")
        img_path = rec.get("image_path") or rec.get("filepath") or rec.get("filename")
        if img_path and not os.path.isabs(img_path):
            img_path = os.path.join(image_root, img_path)
        if not img_path and img_id is not None:
            img_path = _find_image_path(img_id, image_root)
        if not img_path or not os.path.isfile(img_path):
            if img_id is not None:
                img_path = _find_image_path(img_id, image_root)
        if not img_path or not os.path.isfile(img_path):
            continue
        out.append({
            "image_path": os.path.abspath(img_path),
            "image_id": str(img_id) if img_id is not None else image_id_from_path(img_path),
            "captions": caps
        })
    return out

class CaptionValJsonDataset(Dataset):
    def __init__(self, json_path, image_root=""):
        raw = normalize_records(read_json(json_path), image_root)
        self.samples = raw
    
    def __len__(self):  
        return len(self.samples)
    
    def __getitem__(self, idx):
        ex = self.samples[idx]
        caption = ex["captions"][0]
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text",  "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
        return {
            "messages": messages,
            "image_path": ex["image_path"],
            "assistant_text": caption,
            "image_id": ex["image_id"]
        }

class CaptionTrainJsonDataset(Dataset):
    def __init__(self, json_path, image_root=""):
        raw = normalize_records(read_json(json_path), image_root)
        self.samples = raw
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        ex = self.samples[idx]
        caption = random.choice(ex["captions"])
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
        return {
            "messages": messages, 
            "image_path": ex["image_path"], 
            "assistant_text": caption,
            "image_id": ex["image_id"]
        }

class DataCollatorQwenVL:
    def __init__(self, processor):
        self.processor = processor
        self.tok = processor.tokenizer
    
    def __call__(self, batch):
        texts, images, all_messages = [], [], []
        assistant_texts = []
        
        for item in batch:
            messages = item["messages"]
            image_inputs = _extract_images_from_messages(messages)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images.append(image_inputs)
            all_messages.append(messages)
            assistant_texts.append(item.get("assistant_text", ""))
        
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
        
        assistant_header = "<|im_start|>assistant\n"
        assistant_header_ids = self.tok.encode(assistant_header, add_special_tokens=False)
        
        for i in range(len(batch)):
            assistant_start_pos = None
            seq_len = input_ids.shape[1]
            header_len = len(assistant_header_ids)
            
            for j in range(seq_len - header_len + 1):
                match = True
                for k in range(header_len):
                    if input_ids[i, j + k] != assistant_header_ids[k]:
                        match = False
                        break
                
                if match:
                    assistant_start_pos = j + header_len
                    break
            
            if assistant_start_pos is not None:
                labels[i, :assistant_start_pos] = -100
            else:
                print(f"WARNING: Could not find assistant header in sample {i}, using fallback")
                prompt_only = [all_messages[i][0], all_messages[i][1]]
                prompt_text = self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
                enc = self.tok(prompt_text, add_special_tokens=False, return_tensors="pt", padding=False)
                prompt_len = enc["input_ids"].shape[1]
                labels[i, :min(prompt_len, seq_len)] = -100
        
        # DEBUG
        # if True:
        #     for i in range(min(2, len(batch))):
        #         print(f"\n{'='*70}")
        #         print(f"DEBUG SAMPLE {i}")
        #         print(f"{'='*70}")
        #         print(f"Full decoded text:\n{self.tok.decode(input_ids[i])[:500]}...")
        #         print(f"\nTotal tokens: {input_ids.shape[1]}")
                
        #         masked_count = (labels[i] == -100).sum().item()
        #         print(f"Masked tokens: {masked_count}")
                
        #         unmasked_ids = input_ids[i][labels[i] != -100]
        #         if len(unmasked_ids) > 0:
        #             unmasked_text = self.tok.decode(unmasked_ids)
        #             print(f"\nWhat model learns (unmasked):\n{unmasked_text}")
        #         else:
        #             print(f"\nWARNING: Everything is masked!")
                
        #         print(f"\nExpected assistant text:\n{assistant_texts[i]}")
        #         print(f"{'='*70}\n")
        
        batch_out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
        
        for k, v in proc_out.items():
            if k not in batch_out and isinstance(v, torch.Tensor):
                batch_out[k] = v
        
        return batch_out

def load_qwen_with_safe_attn(MODEL_PATH, bnb_config, model_kwargs):
    attn_order = (["flash_attention_2", "sdpa", "eager"] if model_kwargs.get("attn_implementation", None) == "flash_attention_2" else ["sdpa", "eager"])
    last_err = None
    for impl in attn_order:
        try:
            mk = dict(model_kwargs)
            mk["attn_implementation"] = impl
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb_config, **mk)
            return model
        except Exception as e:
            last_err = e
    raise last_err

def do_eval_generate(model, processor, ds, refs_map, num_beams=3, max_new_tokens=128, temperature=0.0, gen_bs=8, use_cache=True):
    import contextlib
    device = next(model.parameters()).device
    prev_attn = getattr(model.config, "attn_implementation", None)
    try:
        if prev_attn == "flash_attention_2":
            model.config.attn_implementation = "eager"
    except Exception:
        pass
    prev_cache = getattr(model.config, "use_cache", None)
    if use_cache:
        model.config.use_cache = True
    preds, refs, img_paths = [], [], []
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        try:
            autocast_ctx = torch.autocast
            autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16}
        except TypeError:
            autocast_ctx = torch.cuda.amp.autocast
            autocast_kwargs = {"dtype": torch.bfloat16}
    else:
        autocast_ctx = contextlib.nullcontext
        autocast_kwargs = {}
    N = len(ds)
    indices = list(range(N))
    for off in range(0, N, gen_bs):
        idx_chunk = indices[off:off + gen_bs]
        chunk = [ds[j] for j in idx_chunk]
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
            batch_images.append(_extract_images_from_messages(messages))
        proc_inputs = processor(text=texts, images=batch_images, padding=True, return_tensors="pt")
        input_lens = proc_inputs["attention_mask"].sum(-1)
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in proc_inputs.items()}
        with torch.inference_mode():
            try:
                with autocast_ctx(**autocast_kwargs):
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=num_beams,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        early_stopping=True,
                        min_new_tokens=MIN_NEW_TOKENS,
                        use_cache=use_cache,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.15,
                        length_penalty=1.0
                    )
            except Exception:
                try:
                    model.config.attn_implementation = "eager"
                except Exception:
                    pass
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    early_stopping=True,
                    min_new_tokens=MIN_NEW_TOKENS,
                    use_cache=use_cache,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.15,
                    length_penalty=1.0
                )
        for j, ex in enumerate(chunk):
            start = int(input_lens[j].item())
            gen_ids = out_ids[j, start:]
            pred = processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
            pred = one_sentence(pred)
            preds.append(pred)
            img_paths.append(ex["image_path"])
            k = ex["image_id"]
            rs = refs_map.get(k) or [ex.get("assistant_text", "")]
            rs = [r.strip() for r in rs if isinstance(r, str)] or [""]
            refs.append(rs)
    if prev_cache is not None:
        model.config.use_cache = prev_cache
    try:
        if prev_attn is not None:
            model.config.attn_implementation = prev_attn
    except Exception:
        pass
    if was_training:
        model.train()
    return preds, refs, img_paths

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

class FullValEachEpochCallback(TrainerCallback):
    def __init__(self, eval_ds, refs_map, processor, num_beams=3, gen_bs=16, max_new_tokens=128, temperature=0.0):
        self.eval_ds = eval_ds
        self.refs_map = refs_map
        self.p = processor
        self.num_beams = num_beams
        self.gen_bs = gen_bs
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._trainer = None
    def set_trainer(self, trainer):
        self._trainer = trainer
    def on_evaluate(self, args, state, control, **kwargs):
        if self._trainer is None or self._trainer.model is None:
            return control
        preds, refs, imgs = do_eval_generate(
            self._trainer.model, self.p, self.eval_ds, self.refs_map,
            num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, temperature=self.temperature, gen_bs=self.gen_bs, use_cache=True
        )
        m = mc.compute_metrics_fast(preds, refs, imgs)
        logs = {f"eval_{k}": float(v) for k, v in m.items() if isinstance(v, (int, float))}
        logs["eval_N_samples"] = float(len(self.eval_ds))
        self._trainer.log(logs)
        try:
            wandb.log(logs, step=state.global_step)
        except Exception:
            pass
        return control

def main():
    if not TRAIN_FILE or not VAL_FILE:
        return
    model_kwargs = {"device_map": "auto"}
    if USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = load_qwen_with_safe_attn(MODEL_PATH, bnb_config, model_kwargs)


    model = prepare_model_for_kbit_training(model)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

    refs_map = {}
    if VAL_FILE:
        raw = read_json(VAL_FILE)
        for rec in raw:
            k = str(rec.get("image_id", ""))
            rs = rec.get("captions") or []
            if k:
                refs_map[k] = [s.strip() for s in rs if isinstance(s, str) and s.strip()]

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    adapter_ok = (
        PEFT_ADAPTER_PATH 
        and os.path.isdir(PEFT_ADAPTER_PATH) 
        and os.path.isfile(os.path.join(PEFT_ADAPTER_PATH, "adapter_config.json"))
    )

    if adapter_ok and HAS_PEFT:
        model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH, is_trainable=True)
        active_adapter = getattr(model, "active_adapter", "default")
        print(f"[PEFT] Loaded adapter from {PEFT_ADAPTER_PATH} (active={active_adapter}) and set to trainable.")
    else:
        from peft import get_peft_model
        model = get_peft_model(model, lora_cfg)
        print("[PEFT] Initialized fresh LoRA adapters.")

    model.enable_input_require_grads()
    train_ds = CaptionTrainJsonDataset(TRAIN_FILE, image_root=IMAGE_ROOT)
    val_ds = CaptionValJsonDataset(VAL_FILE, image_root=IMAGE_ROOT)
    full_eval_cb = FullValEachEpochCallback(eval_ds=val_ds, refs_map=refs_map, processor=processor, num_beams=1, gen_bs=16, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    es_cb = EarlyStopByMetric("eval_CIDEr", greater_is_better=True, patience=5, save_best_dir=os.path.join(OUT_DIR, "best_by_CIDEr"))
    data_collator = DataCollatorQwenVL(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        optim='adamw_bnb_8bit',
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        run_name="qwen2.5-vl_finetune_epoch_eval_fullset_pl_128_continuation",
        eval_strategy="epoch"
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
        callbacks=[full_eval_cb, es_cb]
    )

    full_eval_cb.set_trainer(trainer)
    es_cb.set_trainer(trainer)

    preds, refs, img_paths = do_eval_generate(model, processor, val_ds, refs_map, 
    num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS, gen_bs=8, use_cache=True)

    baseline_metrics = mc.compute_metrics_fast(preds, refs, img_paths)
    print("\n" + "="*70)
    print("BASELINE METRICS (before training):")
    print(json.dumps(baseline_metrics, ensure_ascii=False, indent=2))
    print("="*70 + "\n")

    # Zapisz przykłady
    for i in range(5):
        print(f"Pred: {preds[i]}")
        print(f"Ref:  {refs[i][0]}\n")
        
    trainer.train()
    trainer.model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)

    model.eval()
    preds, refs, img_paths = do_eval_generate(model, processor, val_ds, refs_map, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS, gen_bs=8, use_cache=True)
    metrics = mc.compute_metrics_fast(preds, refs, img_paths)
    mpath = os.path.join(OUT_DIR, "val_metrics.json")
    
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(mpath)
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()