# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 0
# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 1
# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 2
# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 3
# CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 5
# CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 6
# CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 7
# CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 8
# CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 9
# CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 10
# CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 11
# CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 12
# CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 13
# CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 14
# CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 15
# CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 16
# CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 4

#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# CONFIGURATION -----------------------------------------------------
# ------------------------------------------------------------------
ADAPTER=/nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600
SCRIPT=evaluations/eval_visualcloze.py

# Which chunk goes to which GPU.
# Feel free to reshuffle the mapping or add/remove chunks.
declare -A jobs=(
  [0]=0  [1]=0  [2]=0  [3]=0          # GPU-0 → 4 chunks
  [5]=1  [6]=1  [7]=1  [8]=1          # GPU-1 → 4 chunks
  [9]=2  [10]=2 [11]=2 [12]=2 [13]=2  # GPU-2 → 5 chunks
  [14]=3 [15]=3 [16]=3 [4]=3          # GPU-3 → 4 chunks
)

# Maximum *concurrent* processes you’ll allow *per* GPU
MAX_PER_GPU=4
# ------------------------------------------------------------------

# Function that counts running pids for a given GPU
running_for_gpu () {
  local gpu=$1
  jobs -r -p | xargs -r ps -o pid=,args= |
    grep -F "CUDA_VISIBLE_DEVICES=${gpu}" | wc -l
}

echo "Launching VisualCloze evaluations …"
for chunk in "${!jobs[@]}"; do
  gpu=${jobs[$chunk]}

  # Throttle: wait until this GPU is below its concurrency budget
  while [ "$(running_for_gpu "$gpu")" -ge "$MAX_PER_GPU" ]; do
      sleep 5
  done

  echo "▶︎  GPU $gpu  ·  chunk $chunk"
  (
    CUDA_VISIBLE_DEVICES=$gpu \
    python "$SCRIPT" --is_parainj --adapter_path "$ADAPTER" --chunk "$chunk"
  ) &                        # run in the background
done

# Wait until every background job finishes
wait
echo "✅  All chunks finished."