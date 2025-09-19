#!/bin/bash

# Ablation Study Script for Efficiency Analysis
# This script runs different methods (LoRA, PaRa, Injection, Full Fine-tuning) across multiple GPUs

# Check available GPUs
echo "Available GPUs:"
nvidia-smi --list-gpus

# Base configuration
BASE_CMD="accelerate launch --num_processes=1"
MODEL_PATH="Shitao/OmniGen-v1"
JSON_FILE="/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/dreambooth_eval.jsonl"
IMAGE_PATH="/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts"
BASE_RESULTS_DIR="/nvme-data/Komal/documents/results/Ablation"

# Common parameters
COMMON_PARAMS="--model_name_or_path $MODEL_PATH \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --lora_rank 64 \
    --json_file $JSON_FILE \
    --image_path $IMAGE_PATH \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 2000 \
    --epochs 2000 \
    --log_every 1"

# Create timestamp for this ablation run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting ablation study at: $TIMESTAMP"

# Function to run experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local method_name=$2
    local method_flags=$3
    local decomp_method=$4
    
    echo "=== Running $method_name on GPU $gpu_id ==="
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Create results directory for this method
    RESULTS_DIR="$BASE_RESULTS_DIR/${method_name}_${decomp_method}_gpu${gpu_id}_${TIMESTAMP}"
    
    # Build command
    CMD="$BASE_CMD runner_aba.py $COMMON_PARAMS --results_dir $RESULTS_DIR $method_flags"
    
    if [[ ! -z "$decomp_method" ]]; then
        CMD="$CMD --decomposition_method $decomp_method"
    fi
    
    echo "Command: $CMD"
    echo "Results will be saved to: $RESULTS_DIR"
    
    # Run the experiment
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✅ $method_name completed successfully on GPU $gpu_id"
        echo "Results saved to: $RESULTS_DIR"
    else
        echo "❌ $method_name failed on GPU $gpu_id"
    fi
    
    echo "=== Finished $method_name on GPU $gpu_id ==="
    echo ""
}

# Function to run experiments in parallel across GPUs
run_parallel_experiments() {
    echo "🚀 Starting parallel ablation study..."
    
    # Run different methods on different GPUs simultaneously
    
    # GPU 0: LoRA
    run_experiment 0 "lora" "--use_lora" "qr" &
    PID1=$!
    
    # GPU 1: PaRa with QR decomposition
    run_experiment 1 "para_qr" "--use_para" "qr" &
    PID2=$!
    
    # GPU 2: PaRa with SVD decomposition
    run_experiment 2 "para_tsvd" "--use_para" "tsvd" &
    PID3=$!
    
    # GPU 3: Knowledge Injection with QR
    run_experiment 3 "injection_qr" "--use_injection" "qr" &
    PID4=$!
    
    # Wait for all parallel jobs to complete
    echo "Waiting for all experiments to complete..."
    wait $PID1 && echo "✅ LoRA experiment completed"
    wait $PID2 && echo "✅ PaRa QR experiment completed"
    wait $PID3 && echo "✅ PaRa TSVD experiment completed"
    wait $PID4 && echo "✅ Injection QR experiment completed"
    
    echo "🎉 All parallel experiments completed!"
}

# Function to run experiments sequentially (if you prefer or have limited GPUs)
run_sequential_experiments() {
    echo "🔄 Starting sequential ablation study..."
    
    # 1. LoRA baseline
    run_experiment 0 "lora" "--use_lora" ""
    
    # 2. PaRa with different decomposition methods
    run_experiment 0 "para_qr" "--use_para" "qr"
    run_experiment 0 "para_tsvd" "--use_para" "tsvd"
    run_experiment 0 "para_lrmf" "--use_para" "lrmf"
    run_experiment 0 "para_nmf" "--use_para" "nmf"
    run_experiment 0 "para_eigen" "--use_para" "eigen"
    
    # 3. Knowledge Injection with different decomposition methods
    run_experiment 0 "injection_qr" "--use_injection" "qr"
    run_experiment 0 "injection_tsvd" "--use_injection" "tsvd"
    run_experiment 0 "injection_lrmf" "--use_injection" "lrmf"
    
    # 4. Full fine-tuning (if you want to compare)
    # run_experiment 0 "full_finetuning" "" ""
    
    echo "🎉 All sequential experiments completed!"
}

# Function to analyze results
analyze_results() {
    echo "📊 Analyzing results..."
    
    # Create analysis directory
    ANALYSIS_DIR="$BASE_RESULTS_DIR/analysis_$TIMESTAMP"
    mkdir -p $ANALYSIS_DIR
    
    # Collect all efficiency metrics
    echo "Collecting efficiency metrics from all experiments..."
    find $BASE_RESULTS_DIR -name "efficiency_metrics.jsonl" -newer $BASE_RESULTS_DIR 2>/dev/null | while read file; do
        method_dir=$(dirname "$file")
        method_name=$(basename "$method_dir")
        echo "Found metrics for: $method_name"
        cp "$file" "$ANALYSIS_DIR/${method_name}_efficiency_metrics.jsonl"
    done
    
    echo "All efficiency metrics collected in: $ANALYSIS_DIR"
    echo "You can now analyze these files to compare:"
    echo "  - Training time (step_time_ms)"
    echo "  - Memory usage (memory_gb)"
    echo "  - Parameter count (trainable_params)"
    echo "  - Convergence speed (loss over epochs)"
}

# Main execution
echo "🎯 Efficiency Analysis Ablation Study"
echo "======================================"

# Check if runner_aba.py exists
if [ ! -f "runner_aba.py" ]; then
    echo "❌ Error: runner_aba.py not found in current directory"
    echo "Please make sure the training script is available"
    exit 1
fi

# Create base results directory
mkdir -p $BASE_RESULTS_DIR

# Ask user for execution mode
echo "Choose execution mode:"
echo "1) Parallel (uses multiple GPUs simultaneously)"
echo "2) Sequential (uses one GPU for all experiments)"
echo "3) Custom (specify which experiments to run)"
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo "Running parallel experiments..."
        run_parallel_experiments
        ;;
    2)
        echo "Running sequential experiments..."
        run_sequential_experiments
        ;;
    3)
        echo "Custom mode - edit the script to specify which experiments to run"
        # You can customize this section
        run_experiment 0 "injection_qr" "--use_injection" "qr"
        ;;
    *)
        echo "Invalid choice. Running parallel by default..."
        run_parallel_experiments
        ;;
esac

# Analyze results
analyze_results

echo ""
echo "🏁 Ablation study completed!"
echo "📂 Results location: $BASE_RESULTS_DIR"
echo "📊 Analysis files: $BASE_RESULTS_DIR/analysis_$TIMESTAMP"
echo ""
echo "Next steps:"
echo "1. Compare efficiency_metrics.jsonl files"
echo "2. Plot training curves"
echo "3. Analyze memory usage and parameter counts"
echo "4. Compare convergence speeds"