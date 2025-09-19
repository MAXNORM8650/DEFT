
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
    dreambooth_deftQR_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    dreambooth_deft_GFR8_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    dreambooth_deft_GFR4_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    dreambooth_para_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    dreambooth_deftp_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    textual_inversion_sd=dict(bs=1, learning_rate=5e-4, max_train_steps=3000),
)
import os

def model_generator(
    method: Literal[
        "dreambooth_sd",
        "dreambooth_sdxl",
        "dreambooth_lora_sd",
        "dreambooth_lora_sdxl",
        "textual_inversion_sd",
        "dreambooth_deft_sdxl",
        "dreambooth_deftQR_sdxl",
        "dreambooth_deft_GFR8_sdxl"
        "dreambooth_deft_GFR4_sdxl"
        "dreambooth_para_sdxl"
        "dreambooth_deftp_sdxl",
        "textual_inversion_sdxl",
    ],
    output_dir: str | None = None,
    start: int | None = None,
    end: int | None = None,
):
    dreambench_plus = DreamBenchPlus(dir=DREAMBENCH_PLUS_DIR)

    if output_dir is None:
        output_dir = method

    bs = DEFAULT_PARAMS[method]["bs"]
    learning_rate = DEFAULT_PARAMS[method]["learning_rate"]
    max_train_steps = DEFAULT_PARAMS[method]["max_train_steps"]

    if method == "dreambooth_sd":
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
        
        for i, sample in enumerate(dreambench_plus):

            if i >= start and i < end:
                cmd = f"""CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 training_scripts/train_dreambooth.py \
--pretrained_model_name_or_path="{model_name_or_path}" \
--instance_data_dir="{sample.image_path}" \
--output_dir="work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}" \
--instance_prompt="a photo of {sample.subject}" \
--resolution=512 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_steps=99999 \
--push_to_hub \
--seed=42
"""
                os.system(cmd)

    elif method == "dreambooth_lora_sd":
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
        for i, sample in enumerate(dreambench_plus):
            if i >= start and i < end:
                cmd = f"""CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 training_scripts/train_dreambooth_lora.py \
--pretrained_model_name_or_path="{model_name_or_path}" \
--instance_data_dir="{sample.image_path}" \
--output_dir="work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}" \
--instance_prompt="a photo of {sample.subject}" \
--resolution=512 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_epochs=99999 \
--push_to_hub \
--seed=42
"""
                os.system(cmd)

    elif method == "dreambooth_lora_sdxl":
        import os
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        for i, sample in enumerate(dreambench_plus):
            if i >= start and i < end:
                cmd = f"""CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port 29506 training_scripts/train_dreambooth_lora_sdxl.py \
--pretrained_model_name_or_path="{model_name_or_path}"  \
--instance_data_dir="{sample.image_path}" \
--pretrained_vae_model_name_or_path="{vae_name_or_path}" \
--output_dir="work_dirs/dreambench_plus/{output_dir}_progress/{sample.collection_id}" \
--mixed_precision="fp16" \
--instance_prompt="a photo of {sample.subject}" \
--resolution=1024 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_epochs=99999 \
--checkpointing_steps=50 \
--seed=42
"""
                os.system(cmd)

    elif method == "dreambooth_deft_sdxl":
        import os
        import torch
        import concurrent.futures
        
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs available")
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Number of processes per GPU
        processes_per_gpu = 2  # Each GPU can handle 2 samples simultaneously
        total_parallel_processes = num_gpus * processes_per_gpu
        print(f"Will run {total_parallel_processes} processes in parallel")
        
        # Filter samples based on start and end indices
        samples_to_process = [sample for i, sample in enumerate(dreambench_plus) if i >= start and i < end]
        print(f"Need to process {len(samples_to_process)} samples")
        
        def process_sample(args):
            process_id, sample = args
            gpu_id = process_id % num_gpus  # Determine which GPU to use
            
            print(f"Processing sample {sample.collection_id} on GPU {gpu_id} (process ID: {process_id})")
            
            # Set different output directories for parallel processes on the same GPU to avoid conflicts
            output_dir = f"work_dirs/dreambench_plus/{method}/{sample.collection_id}"
            
            cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training_scripts/main_deft_sdxl.py \
            --pretrained_model_name_or_path="{model_name_or_path}" \
            --instance_data_dir="{sample.image_path}" \
            --pretrained_vae_model_name_or_path="{vae_name_or_path}" \
            --output_dir="{output_dir}" \
            --mixed_precision="fp16" \
            --instance_prompt="a photo of {sample.subject}" \
            --resolution=1024 \
            --train_batch_size={bs} \
            --gradient_accumulation_steps=1 \
            --learning_rate={learning_rate} \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={max_train_steps} \
            --validation_epochs=99999 \
            --checkpointing_steps=50 \
            --seed=42"""
            
            return os.system(cmd)
        
        # Process samples in parallel with multiple processes per GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_parallel_processes) as executor:
            # Assign process IDs to each sample
            process_sample_pairs = [(i, sample) for i, sample in enumerate(samples_to_process)]
            
            # Execute in batches if there are more samples than available parallel processes
            for i in range(0, len(process_sample_pairs), total_parallel_processes):
                batch = process_sample_pairs[i:i + total_parallel_processes]
                print(f"Processing batch of {len(batch)} samples")
                results = list(executor.map(process_sample, batch))
                
                # Check results
                failed_samples = [batch[j][1].collection_id for j, res in enumerate(results) if res != 0]
                if failed_samples:
                    print(f"Warning: Failed to process samples: {failed_samples}")
        
        print("All samples processed!")


    elif method == "dreambooth_deftQR_sdxl":
        import os
        import torch
        import concurrent.futures
        
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs available")
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Number of processes per GPU
        processes_per_gpu = 2  # Each GPU can handle 2 samples simultaneously
        total_parallel_processes = num_gpus * processes_per_gpu
        print(f"Will run {total_parallel_processes} processes in parallel")
        
        # Filter samples based on start and end indices
        samples_to_process = [sample for i, sample in enumerate(dreambench_plus) if i >= start and i < end]
        print(f"Need to process {len(samples_to_process)} samples")
        
        def process_sample(args):
            process_id, sample = args
            gpu_id = process_id % num_gpus  # Determine which GPU to use
            
            print(f"Processing sample {sample.collection_id} on GPU {gpu_id} (process ID: {process_id})")
            
            # Set different output directories for parallel processes on the same GPU to avoid conflicts
            output_dir = f"work_dirs/dreambench_plus/{method}/{sample.collection_id}"
            
            cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training_scripts/main_deft_sdxl.py \
            --pretrained_model_name_or_path="{model_name_or_path}" \
            --instance_data_dir="{sample.image_path}" \
            --pretrained_vae_model_name_or_path="{vae_name_or_path}" \
            --output_dir="{output_dir}" \
            --mixed_precision="fp16" \
            --instance_prompt="a photo of {sample.subject}" \
            --resolution=1024 \
            --train_batch_size={bs} \
            --gradient_accumulation_steps=1 \
            --learning_rate={learning_rate} \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={max_train_steps} \
            --validation_epochs=99999 \
            --seed=42"""
            
            return os.system(cmd)
        
        # Process samples in parallel with multiple processes per GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_parallel_processes) as executor:
            # Assign process IDs to each sample
            process_sample_pairs = [(i, sample) for i, sample in enumerate(samples_to_process)]
            
            # Execute in batches if there are more samples than available parallel processes
            for i in range(0, len(process_sample_pairs), total_parallel_processes):
                batch = process_sample_pairs[i:i + total_parallel_processes]
                print(f"Processing batch of {len(batch)} samples")
                results = list(executor.map(process_sample, batch))
                
                # Check results
                failed_samples = [batch[j][1].collection_id for j, res in enumerate(results) if res != 0]
                if failed_samples:
                    print(f"Warning: Failed to process samples: {failed_samples}")
        
        print("All samples processed!")
    elif method == "dreambooth_deft_GFR8_sdxl":
        import os
        import torch
        import concurrent.futures
        
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs available")
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Number of processes per GPU
        processes_per_gpu = 2  # Each GPU can handle 2 samples simultaneously
        total_parallel_processes = num_gpus * processes_per_gpu
        print(f"Will run {total_parallel_processes} processes in parallel")
        
        # Filter samples based on start and end indices
        samples_to_process = [sample for i, sample in enumerate(dreambench_plus) if i >= start and i < end]
        print(f"Need to process {len(samples_to_process)} samples")
        
        def process_sample(args):
            process_id, sample = args
            gpu_id = process_id % num_gpus  # Determine which GPU to use
            
            print(f"Processing sample {sample.collection_id} on GPU {gpu_id} (process ID: {process_id})")
            
            # Set different output directories for parallel processes on the same GPU to avoid conflicts
            output_dir = f"work_dirs/dreambench_plus/{method}/{sample.collection_id}"
            
            cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training_scripts/main_deft_sdxl.py \
            --pretrained_model_name_or_path="{model_name_or_path}" \
            --instance_data_dir="{sample.image_path}" \
            --pretrained_vae_model_name_or_path="{vae_name_or_path}" \
            --output_dir="{output_dir}" \
            --mixed_precision="fp16" \
            --instance_prompt="a photo of {sample.subject}" \
            --resolution=1024 \
            --train_batch_size={bs} \
            --gradient_accumulation_steps=1 \
            --learning_rate={learning_rate} \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={max_train_steps} \
            --validation_epochs=99999 \
            --seed=42"""
            
            return os.system(cmd)
        
        # Process samples in parallel with multiple processes per GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_parallel_processes) as executor:
            # Assign process IDs to each sample
            process_sample_pairs = [(i, sample) for i, sample in enumerate(samples_to_process)]
            
            # Execute in batches if there are more samples than available parallel processes
            for i in range(0, len(process_sample_pairs), total_parallel_processes):
                batch = process_sample_pairs[i:i + total_parallel_processes]
                print(f"Processing batch of {len(batch)} samples")
                results = list(executor.map(process_sample, batch))
                
                # Check results
                failed_samples = [batch[j][1].collection_id for j, res in enumerate(results) if res != 0]
                if failed_samples:
                    print(f"Warning: Failed to process samples: {failed_samples}")
        
        print("All samples processed!")

    elif method == "dreambooth_deft_GFR4_sdxl":
        import os
        import torch
        import concurrent.futures
        
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs available")
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Number of processes per GPU
        processes_per_gpu = 2  # Each GPU can handle 2 samples simultaneously
        total_parallel_processes = num_gpus * processes_per_gpu
        print(f"Will run {total_parallel_processes} processes in parallel")
        
        # Filter samples based on start and end indices
        samples_to_process = [sample for i, sample in enumerate(dreambench_plus) if i >= start and i < end]
        print(f"Need to process {len(samples_to_process)} samples")
        
        def process_sample(args):
            process_id, sample = args
            gpu_id = process_id % num_gpus  # Determine which GPU to use
            
            print(f"Processing sample {sample.collection_id} on GPU {gpu_id} (process ID: {process_id})")
            
            # Set different output directories for parallel processes on the same GPU to avoid conflicts
            output_dir = f"work_dirs/dreambench_plus/{method}_progress/{sample.collection_id}"
            
            cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training_scripts/main_deft_sdxl.py \
            --pretrained_model_name_or_path="{model_name_or_path}" \
            --instance_data_dir="{sample.image_path}" \
            --pretrained_vae_model_name_or_path="{vae_name_or_path}" \
            --output_dir="{output_dir}" \
            --mixed_precision="fp16" \
            --instance_prompt="a photo of {sample.subject}" \
            --resolution=1024 \
            --train_batch_size={bs} \
            --gradient_accumulation_steps=1 \
            --learning_rate={learning_rate} \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={max_train_steps} \
            --validation_epochs=99999 \
            --checkpointing_steps=50 \
            --rank=4 \
            --seed=42"""
            
            return os.system(cmd)
        
        # Process samples in parallel with multiple processes per GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_parallel_processes) as executor:
            # Assign process IDs to each sample
            process_sample_pairs = [(i, sample) for i, sample in enumerate(samples_to_process)]
            
            # Execute in batches if there are more samples than available parallel processes
            for i in range(0, len(process_sample_pairs), total_parallel_processes):
                batch = process_sample_pairs[i:i + total_parallel_processes]
                print(f"Processing batch of {len(batch)} samples")
                results = list(executor.map(process_sample, batch))
                
                # Check results
                failed_samples = [batch[j][1].collection_id for j, res in enumerate(results) if res != 0]
                if failed_samples:
                    print(f"Warning: Failed to process samples: {failed_samples}")
        
        print("All samples processed!")

    elif method == "dreambooth_para_sdxl":
        import os
        import torch
        import concurrent.futures
        
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs available")
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Number of processes per GPU
        processes_per_gpu = 2  # Each GPU can handle 2 samples simultaneously
        total_parallel_processes = num_gpus * processes_per_gpu
        print(f"Will run {total_parallel_processes} processes in parallel")
        
        # Filter samples based on start and end indices
        samples_to_process = [sample for i, sample in enumerate(dreambench_plus) if i >= start and i < end]
        print(f"Need to process {len(samples_to_process)} samples")
        
        def process_sample(args):
            process_id, sample = args
            gpu_id = process_id % num_gpus  # Determine which GPU to use
            
            print(f"Processing sample {sample.collection_id} on GPU {gpu_id} (process ID: {process_id})")
            
            # Set different output directories for parallel processes on the same GPU to avoid conflicts
            output_dir = f"work_dirs/dreambench_plus/{method}_prograss/{sample.collection_id}"
            
            cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training_scripts/main_deft_sdxl.py \
            --pretrained_model_name_or_path="{model_name_or_path}" \
            --instance_data_dir="{sample.image_path}" \
            --pretrained_vae_model_name_or_path="{vae_name_or_path}" \
            --output_dir="{output_dir}" \
            --mixed_precision="fp16" \
            --instance_prompt="a photo of {sample.subject}" \
            --resolution=1024 \
            --train_batch_size={bs} \
            --gradient_accumulation_steps=1 \
            --learning_rate={learning_rate} \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={max_train_steps} \
            --validation_epochs=99999 \
            --checkpointing_steps=50 \
            --rank=4 \
            --Rgate=0 \
            --ortho=True \
            --seed=42"""
            
            return os.system(cmd)
        
        # Process samples in parallel with multiple processes per GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_parallel_processes) as executor:
            # Assign process IDs to each sample
            process_sample_pairs = [(i, sample) for i, sample in enumerate(samples_to_process)]
            
            # Execute in batches if there are more samples than available parallel processes
            for i in range(0, len(process_sample_pairs), total_parallel_processes):
                batch = process_sample_pairs[i:i + total_parallel_processes]
                print(f"Processing batch of {len(batch)} samples")
                results = list(executor.map(process_sample, batch))
                
                # Check results
                failed_samples = [batch[j][1].collection_id for j, res in enumerate(results) if res != 0]
                if failed_samples:
                    print(f"Warning: Failed to process samples: {failed_samples}")
        
        print("All samples processed!")

    elif method == "dreambooth_deftp_sdxl_old":
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        # breakpoint()
        for i, sample in enumerate(dreambench_plus):
            output_dir = f"work_dirs/dreambench_plus/dreambooth_deftp_sdxl/{sample.collection_id}"

            output_dir = Path(output_dir)
            expected_final = output_dir / f"checkpoint-{max_train_steps}" / "project_replace.pt"

            if expected_final.exists():
                print(f"[exit] found — skipping training: {expected_final} exists.")
                # print(f"Training already completed")
                continue # or exit() if you want to stop the script entirely
            else:
                print(f"Training starting: {expected_final} does not exists.")


            if i >= start and i < end:
                cmd = f"""CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_port 29506 training_scripts/main_deft_sdxl.py \
--pretrained_model_name_or_path="{model_name_or_path}"  \
--instance_data_dir="{sample.image_path}" \
--pretrained_vae_model_name_or_path="{vae_name_or_path}" \
--output_dir="work_dirs/dreambench_plus/dreambooth_deftp_sdxl/{sample.collection_id}" \
--mixed_precision="fp16" \
--instance_prompt="a photo of {sample.subject}" \
--resolution=1024 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_epochs=99999 \
--seed=42
"""
                os.system(cmd)
    elif method == "dreambooth_deftp_sdxl":
        # ------------------------------------------------------------------ settings
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path   = "madebyollin/sdxl-vae-fp16-fix"
        GPUS               = ["1", "2"]
        MASTER_PORT        = [29515, 29516, 29517]
        # ---------------------------------------------------------------------------

        import itertools, subprocess, logging, os, time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

        # ---------- helpers --------------------------------------------------------
        def paths(sample):
            root = Path(f"work_dirs/dreambench_plus/dreambooth_deftp_sdxl/{sample.collection_id}")
            final_ckpt = root / f"checkpoint-{max_train_steps}" / "project_replace.pt"
            lock_file  = root / ".training.lock"
            return root, final_ckpt, lock_file

        def needs_training(sample) -> bool:
            root, final_ckpt, lock_file = paths(sample)
            if final_ckpt.is_file():
                logging.info(f"[skip✅] {sample.collection_id} already finished.")
                return False
            if lock_file.is_file():
                # Another job **claims** the directory ‑> skip
                logging.info(f"[skip⏳] {sample.collection_id} in progress elsewhere.")
                return False
            return True

        # --- create lock atomically, or give up ------------------------------------
        def try_lock(lock_path: Path) -> bool:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(lock_path, "x") as f:          # 'x' == O_CREAT|O_EXCL
                    f.write(f"{os.getpid()}\n{time.time():.0f}")
                return True
            except FileExistsError:
                return False

        def launch(sample, gpu_id: str):
            root, _, lock_file = paths(sample)
            if not try_lock(lock_file):
                logging.info(f"[race] {sample.collection_id} lock held by another; skipping.")
                return 0  # someone else is (or was) working

            out_dir = root
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_id} "
                f"python -m torch.distributed.run --nproc_per_node=1 "
                f"--master_port {MASTER_PORT[int(gpu_id)]} "
                f"training_scripts/main_deft_sdxl.py "
                f'--pretrained_model_name_or_path="{model_name_or_path}" '
                f'--pretrained_vae_model_name_or_path="{vae_name_or_path}" '
                f'--instance_data_dir="{sample.image_path}" '
                f'--output_dir="{out_dir}" '
                f'--instance_prompt="a photo of {sample.subject}" '
                f"--mixed_precision=fp16 "
                f"--resolution=1024 "
                f"--train_batch_size={bs} "
                f"--gradient_accumulation_steps=1 "
                f"--learning_rate={learning_rate} "
                f"--lr_scheduler=constant "
                f"--lr_warmup_steps=0 "
                f"--max_train_steps={max_train_steps} "
                f"--validation_epochs=99999 "
                f"--seed=42 "
            )

            logging.info(f"[run▶] {sample.collection_id} on GPU {gpu_id}")
            rc = subprocess.run(cmd, shell=True).returncode

            # Clean up the lock no matter what
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            return rc
        # ---------------------------------------------------------------------------

        subset = dreambench_plus[slice(start, end)]
        todo   = [s for s in subset if needs_training(s)]
        if not todo:
            logging.info("Nothing to do — all samples either finished or running.")
            return

        gpu_cycle = itertools.cycle(GPUS)
        with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            fut2sample = {pool.submit(launch, s, next(gpu_cycle)): s for s in todo}
            for fut in as_completed(fut2sample):
                s = fut2sample[fut]
                try:
                    rc = fut.result()
                    msg = "✅ done" if rc == 0 else f"❌ exit {rc}"
                    logging.info(f"[{msg}] {s.collection_id}")
                except Exception as e:
                    logging.exception(f"[error] {s.collection_id}: {e}")
    elif method == "textual_inversion_sd":
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
        for i, sample in enumerate(dreambench_plus):
            _class_single_token = sample.subject.split(" ")[0]
            if "style" in sample.image_path:
                learnable_property = "style"
            else:
                learnable_property = "object"
            if i >= start and i < end:
                cmd = f"""CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 training_scripts/textual_inversion.py \
--pretrained_model_name_or_path="{model_name_or_path}" \
--train_data_dir="{sample.image_path}" \
--learnable_property="{learnable_property}" \
--placeholder_token="<sks>" \
--initializer_token="{_class_single_token}" \
--resolution=512 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--max_train_steps={max_train_steps} \
--learning_rate={learning_rate} \
--scale_lr \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--output_dir="work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}"
"""
                os.system(cmd)


if __name__ == "__main__":
    fire.Fire(model_generator)
