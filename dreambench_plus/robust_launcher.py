import contextlib, socket, random, subprocess, logging, os, time, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path




from typing import Literal

import fire
from pathlib import Path
from dreambench_plus.constants import DREAMBENCH_PLUS_DIR
from dreambench_plus.dreambench_plus_dataset import DreamBenchPlus

DEFAULT_PARAMS = dict(
    dreambooth_sd=dict(bs=1, learning_rate=2.5e-6, max_train_steps=250),
    dreambooth_lora_sd=dict(bs=1, learning_rate=1e-4, max_train_steps=100),
    dreambooth_lora_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    dreambooth_deft_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    dreambooth_deftp_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    textual_inversion_sd=dict(bs=1, learning_rate=5e-4, max_train_steps=3000),
)
import os
model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
method = "dreambooth_deft_sdxl"
bs = DEFAULT_PARAMS[method]["bs"]
learning_rate = DEFAULT_PARAMS[method]["learning_rate"]
max_train_steps = DEFAULT_PARAMS[method]["max_train_steps"]

# … settings …
GPUS = ["0","1", "2", "3"]
SLOTS_PER_GPU = 2                      # 1 job per card here
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)

# ───────────────────────────────── helpers
def free_port() -> int:
    """Ask the OS for a free TCP port."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def paths(sample):
    root = Path(f"work_dirs/dreambench_plus/dreambooth_deftp_sdxl/{sample.collection_id}")
    final_ckpt = root / f"checkpoint-{max_train_steps}" / "project_replace.pt"
    lock_file  = root / ".training.lock"
    log_file   = LOG_DIR / f"{sample.collection_id}.log"
    return root, final_ckpt, lock_file, log_file

def try_lock(path: Path, ttl_hr: int = 24) -> bool:
    now = time.time()
    if path.exists():
        # stale?
        mtime = path.stat().st_mtime
        if (now - mtime) / 3600 > ttl_hr:
            path.unlink(missing_ok=True)
        else:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(f"{os.getpid()}\n{now:.0f}")
    return True

def launch(sample, gpu_id: str):
    root, _, lock_file, log_file = paths(sample)
    if not try_lock(lock_file):
        logging.info(f"[lock] {sample.collection_id} busy elsewhere.")
        return 0

    port = free_port()
    cmd = f"""
CUDA_VISIBLE_DEVICES={gpu_id} \
python training_scripts/main_deft_sdxl.py \
  --pretrained_model_name_or_path="{model_name_or_path}" \
  --pretrained_vae_model_name_or_path="{vae_name_or_path}" \
  --instance_data_dir="{sample.image_path}" \
  --output_dir="{root}" \
  --instance_prompt="a photo of {sample.subject}" \
  --mixed_precision=fp16 \
  --resolution=1024 \
  --train_batch_size={bs} \
  --gradient_accumulation_steps=1 \
  --learning_rate={learning_rate} \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps={max_train_steps} \
  --validation_epochs=99999 \
  --seed=42
"""
    # If you really need torchrun, prepend:
    # cmd = f"torchrun --nproc_per_node=1 --master_port={port} " + cmd

    logging.info(f"[run▶] {sample.collection_id} gpu={gpu_id} port={port}")
    with open(log_file, "w") as lf:
        rc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT).returncode

    lock_file.unlink(missing_ok=True)
    return rc