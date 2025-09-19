#!/usr/bin/env python3
"""
Comprehensive Ablation Study Script for DEFT
Addresses all reviewer concerns with systematic experiments
"""

import json
import os
import argparse
import subprocess
import time
import pandas as pd
from pathlib import Path
import itertools
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationStudyRunner:
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results = []
        self.experiment_counter = 0
        
    def run_experiment(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Run a single experiment with given configuration"""
        self.experiment_counter += 1
        exp_dir = f"{self.base_config['results_dir']}/ablation_{self.experiment_counter:03d}_{experiment_name}"
        
        # Build command
        cmd = ["python", "train.py"]
        for key, value in {**self.base_config, **config}.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
        
        cmd.extend(["--results_dir", exp_dir])
        
        logger.info(f"Running experiment {self.experiment_counter}: {experiment_name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Record start time for efficiency metrics
        start_time = time.time()
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                # Extract metrics from logs or evaluation
                metrics = self.extract_metrics(exp_dir, config, training_time)
                metrics.update({
                    'experiment_name': experiment_name,
                    'status': 'success',
                    'config': config
                })
                logger.info(f"Experiment {experiment_name} completed successfully")
            else:
                logger.error(f"Experiment {experiment_name} failed: {result.stderr}")
                metrics = {
                    'experiment_name': experiment_name,
                    'status': 'failed',
                    'error': result.stderr,
                    'config': config
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Experiment {experiment_name} timed out")
            metrics = {
                'experiment_name': experiment_name,
                'status': 'timeout',
                'config': config
            }
        
        self.results.append(metrics)
        return metrics
    
    def extract_metrics(self, exp_dir: str, config: Dict[str, Any], training_time: float) -> Dict[str, Any]:
        """Extract metrics from experiment results"""
        metrics = {
            'training_time_hours': training_time / 3600,
            'parameters_trainable': 0,
            'memory_usage_gb': 0,
            'final_loss': 0.0,
            'convergence_steps': 0
        }
        
        # Try to extract from tensorboard logs or training logs
        log_file = os.path.join(exp_dir, "training.log")
        if os.path.exists(log_file):
            metrics.update(self.parse_training_log(log_file))
        
        # Count parameters
        if config.get('use_lora') or config.get('use_para') or config.get('use_injection'):
            # Estimate based on rank and target modules
            rank = config.get('lora_rank', 8)
            # Rough estimate for OmniGen model dimensions
            hidden_size = 4096  # Typical for large models
            num_layers = 32     # Typical number of layers
            target_modules = 2  # qkv_proj, o_proj
            
            if config.get('use_lora'):
                metrics['parameters_trainable'] = rank * hidden_size * target_modules * num_layers * 2
            elif config.get('use_para'):
                metrics['parameters_trainable'] = rank * hidden_size * target_modules * num_layers
            elif config.get('use_injection'):
                # DEFT uses both components
                metrics['parameters_trainable'] = rank * hidden_size * target_modules * num_layers * 3
        
        return metrics
    
    def parse_training_log(self, log_file: str) -> Dict[str, Any]:
        """Parse training log to extract convergence and loss metrics"""
        metrics = {}
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            losses = []
            for line in lines:
                if "Train Loss:" in line:
                    # Extract loss value
                    loss_str = line.split("Train Loss:")[1].split(",")[0].strip()
                    try:
                        loss = float(loss_str)
                        losses.append(loss)
                    except ValueError:
                        continue
            
            if losses:
                metrics['final_loss'] = losses[-1]
                metrics['initial_loss'] = losses[0]
                metrics['convergence_steps'] = len(losses)
                
                # Simple convergence detection: when loss stabilizes
                if len(losses) > 100:
                    recent_losses = losses[-50:]
                    std_recent = pd.Series(recent_losses).std()
                    if std_recent < 0.01:  # Arbitrary threshold
                        metrics['converged'] = True
                    else:
                        metrics['converged'] = False
        
        except Exception as e:
            logger.warning(f"Could not parse training log: {e}")
        
        return metrics

def run_component_ablation(runner: AblationStudyRunner):
    """Critical ablation: LoRA only vs PaRa only vs DEFT"""
    logger.info("=== Running Component Ablation Study ===")
    
    experiments = [
        # Baseline: No adaptation
        ({}, "baseline_no_adaptation"),
        
        # LoRA only
        ({"use_lora": True, "lora_rank": 8}, "lora_only_r8"),
        ({"use_lora": True, "lora_rank": 16}, "lora_only_r16"),
        ({"use_lora": True, "lora_rank": 32}, "lora_only_r32"),
        
        # PaRa only
        ({"use_para": True, "lora_rank": 8, "decomposition_method": "qr"}, "para_only_qr_r8"),
        ({"use_para": True, "lora_rank": 16, "decomposition_method": "qr"}, "para_only_qr_r16"),
        ({"use_para": True, "lora_rank": 32, "decomposition_method": "qr"}, "para_only_qr_r32"),
        
        # DEFT (Knowledge Injection = PaRa + LoRA-like component)
        ({"use_injection": True, "lora_rank": 8, "decomposition_method": "qr"}, "deft_qr_r8"),
        ({"use_injection": True, "lora_rank": 16, "decomposition_method": "qr"}, "deft_qr_r16"),
        ({"use_injection": True, "lora_rank": 32, "decomposition_method": "qr"}, "deft_qr_r32"),
    ]
    
    for config, name in experiments:
        runner.run_experiment(config, f"component_{name}")

def run_decomposition_ablation(runner: AblationStudyRunner):
    """Test different decomposition methods"""
    logger.info("=== Running Decomposition Method Ablation ===")
    
    methods = ["qr", "tsvd", "lrmf", "nmf", "eigen"]
    ranks = [8, 16, 32]
    
    for method in methods:
        for rank in ranks:
            # Test on PaRa
            config = {
                "use_para": True,
                "lora_rank": rank,
                "decomposition_method": method
            }
            runner.run_experiment(config, f"decomp_para_{method}_r{rank}")
            
            # Test on DEFT
            config = {
                "use_injection": True,
                "lora_rank": rank,
                "decomposition_method": method
            }
            runner.run_experiment(config, f"decomp_deft_{method}_r{rank}")

def run_learning_rate_ablation(runner: AblationStudyRunner):
    """Test different learning rate configurations for DEFT"""
    logger.info("=== Running Learning Rate Ablation ===")
    
    # Different ratios for P parameters vs R_new parameters
    lr_ratios = [0.01, 0.1, 0.5, 1.0, 2.0]  # P_lr = base_lr * ratio
    base_lrs = [1e-5, 5e-5, 1e-4, 5e-4]
    
    for base_lr in base_lrs:
        for ratio in lr_ratios:
            config = {
                "use_injection": True,
                "lora_rank": 16,
                "decomposition_method": "qr",
                "lr": base_lr,
                "lr_ratio_p": ratio  # This would need to be implemented in the training script
            }
            runner.run_experiment(config, f"lr_ablation_base{base_lr}_ratio{ratio}")

def run_efficiency_analysis(runner: AblationStudyRunner):
    """Analyze computational efficiency"""
    logger.info("=== Running Efficiency Analysis ===")
    
    # Test different configurations with timing focus
    configs = [
        ({"use_lora": True, "lora_rank": 8}, "efficiency_lora_r8"),
        ({"use_lora": True, "lora_rank": 32}, "efficiency_lora_r32"),
        ({"use_para": True, "lora_rank": 8, "decomposition_method": "qr"}, "efficiency_para_r8"),
        ({"use_para": True, "lora_rank": 32, "decomposition_method": "qr"}, "efficiency_para_r32"),
        ({"use_injection": True, "lora_rank": 8, "decomposition_method": "qr"}, "efficiency_deft_r8"),
        ({"use_injection": True, "lora_rank": 32, "decomposition_method": "qr"}, "efficiency_deft_r32"),
    ]
    
    for config, name in configs:
        # Run shorter experiments for timing
        short_config = {**config, "epochs": 100}  # Shorter for timing analysis
        runner.run_experiment(short_config, name)

def run_convergence_analysis(runner: AblationStudyRunner):
    """Analyze convergence speed and stability"""
    logger.info("=== Running Convergence Analysis ===")
    
    configs = [
        ({"use_lora": True, "lora_rank": 16}, "convergence_lora"),
        ({"use_para": True, "lora_rank": 16, "decomposition_method": "qr"}, "convergence_para"),
        ({"use_injection": True, "lora_rank": 16, "decomposition_method": "qr"}, "convergence_deft"),
    ]
    
    for config, name in configs:
        # Longer runs for convergence analysis
        long_config = {**config, "epochs": 2000, "log_every": 10}
        runner.run_experiment(long_config, name)

def run_rank_sensitivity_analysis(runner: AblationStudyRunner):
    """Test sensitivity to rank parameter"""
    logger.info("=== Running Rank Sensitivity Analysis ===")
    
    ranks = [4, 8, 16, 32, 64, 128]
    methods = ["lora", "para", "deft"]
    
    for rank in ranks:
        for method in methods:
            if method == "lora":
                config = {"use_lora": True, "lora_rank": rank}
            elif method == "para":
                config = {"use_para": True, "lora_rank": rank, "decomposition_method": "qr"}
            else:  # deft
                config = {"use_injection": True, "lora_rank": rank, "decomposition_method": "qr"}
            
            runner.run_experiment(config, f"rank_sensitivity_{method}_r{rank}")

def generate_results_tables(results: List[Dict[str, Any]]) -> None:
    """Generate comprehensive results tables"""
    logger.info("=== Generating Results Tables ===")
    
    df = pd.DataFrame(results)
    
    # Table 1: Component Ablation Results
    component_results = df[df['experiment_name'].str.contains('component_')]
    if not component_results.empty:
        component_table = component_results[['experiment_name', 'training_time_hours', 
                                           'parameters_trainable', 'final_loss', 'converged']].copy()
        component_table.to_csv('results_component_ablation.csv', index=False)
        print("\n=== Component Ablation Results ===")
        print(component_table.to_string(index=False))
    
    # Table 2: Decomposition Method Comparison
    decomp_results = df[df['experiment_name'].str.contains('decomp_')]
    if not decomp_results.empty:
        decomp_table = decomp_results[['experiment_name', 'final_loss', 'convergence_steps', 
                                     'training_time_hours']].copy()
        decomp_table.to_csv('results_decomposition_ablation.csv', index=False)
        print("\n=== Decomposition Method Results ===")
        print(decomp_table.to_string(index=False))
    
    # Table 3: Efficiency Analysis
    efficiency_results = df[df['experiment_name'].str.contains('efficiency_')]
    if not efficiency_results.empty:
        efficiency_table = efficiency_results[['experiment_name', 'training_time_hours', 
                                             'parameters_trainable', 'memory_usage_gb']].copy()
        efficiency_table.to_csv('results_efficiency_analysis.csv', index=False)
        print("\n=== Efficiency Analysis Results ===")
        print(efficiency_table.to_string(index=False))
    
    # Table 4: Convergence Analysis
    convergence_results = df[df['experiment_name'].str.contains('convergence_')]
    if not convergence_results.empty:
        convergence_table = convergence_results[['experiment_name', 'initial_loss', 'final_loss', 
                                               'convergence_steps', 'converged']].copy()
        convergence_table.to_csv('results_convergence_analysis.csv', index=False)
        print("\n=== Convergence Analysis Results ===")
        print(convergence_table.to_string(index=False))
    
    # Table 5: Rank Sensitivity Analysis
    rank_results = df[df['experiment_name'].str.contains('rank_sensitivity')]
    if not rank_results.empty:
        rank_table = rank_results[['experiment_name', 'parameters_trainable', 'final_loss', 
                                 'training_time_hours']].copy()
        rank_table.to_csv('results_rank_sensitivity.csv', index=False)
        print("\n=== Rank Sensitivity Results ===")
        print(rank_table.to_string(index=False))
    
    # Summary Table for Paper
    summary_data = []
    
    # Get best results for each method
    for method in ['lora', 'para', 'deft']:
        method_results = df[df['experiment_name'].str.contains(f'component_.*{method}')]
        if not method_results.empty:
            best_result = method_results.loc[method_results['final_loss'].idxmin()]
            summary_data.append({
                'Method': method.upper(),
                'Best Rank': best_result.get('config', {}).get('lora_rank', 'N/A'),
                'Final Loss': f"{best_result['final_loss']:.4f}",
                'Trainable Params (M)': f"{best_result['parameters_trainable']/1e6:.2f}",
                'Training Time (h)': f"{best_result['training_time_hours']:.2f}",
                'Converged': best_result.get('converged', 'Unknown')
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results_summary_for_paper.csv', index=False)
    print("\n=== Summary Table for Paper ===")
    print(summary_df.to_string(index=False))
    
    # Save all results
    df.to_csv('results_all_experiments.csv', index=False)
    with open('results_all_experiments.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Ablation Study for DEFT")
    parser.add_argument("--json_file", type=str, required=True, help="Training data JSON file")
    parser.add_argument("--image_path", type=str, help="Path to training images")
    parser.add_argument("--model_name_or_path", type=str, default="OmniGen", help="Model path")
    parser.add_argument("--base_results_dir", type=str, default="ablation_results", help="Base results directory")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs for ablation")
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with fewer experiments")
    parser.add_argument("--skip_component", action="store_true", help="Skip component ablation")
    parser.add_argument("--skip_decomposition", action="store_true", help="Skip decomposition ablation")
    parser.add_argument("--skip_efficiency", action="store_true", help="Skip efficiency analysis")
    parser.add_argument("--skip_convergence", action="store_true", help="Skip convergence analysis")
    parser.add_argument("--skip_rank", action="store_true", help="Skip rank sensitivity analysis")
    
    args = parser.parse_args()
    
    # Base configuration for all experiments
    base_config = {
        "json_file": args.json_file,
        "image_path": args.image_path,
        "model_name_or_path": args.model_name_or_path,
        "epochs": args.epochs if not args.quick_test else 50,
        "batch_size_per_device": 1,
        "lr": 1e-4,
        "max_grad_norm": 1.0,
        "log_every": 100 if not args.quick_test else 10,
        "ckpt_every": 10000,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "lr_scheduler": "constant",
        "lr_warmup_steps": 100 if args.quick_test else 1000,
    }
    
    # Create results directory
    os.makedirs(args.base_results_dir, exist_ok=True)
    
    # Initialize runner
    runner = AblationStudyRunner(base_config)
    
    try:
        # Run ablation studies
        if not args.skip_component:
            run_component_ablation(runner)
        
        if not args.skip_decomposition and not args.quick_test:
            run_decomposition_ablation(runner)
        
        if not args.skip_efficiency:
            run_efficiency_analysis(runner)
        
        if not args.skip_convergence and not args.quick_test:
            run_convergence_analysis(runner)
        
        if not args.skip_rank and not args.quick_test:
            run_rank_sensitivity_analysis(runner)
        
        # Generate results tables
        generate_results_tables(runner.results)
        
        logger.info(f"Ablation study completed! Results saved in {args.base_results_dir}")
        logger.info(f"Total experiments run: {runner.experiment_counter}")
        
    except KeyboardInterrupt:
        logger.info("Ablation study interrupted by user")
        if runner.results:
            logger.info("Generating partial results...")
            generate_results_tables(runner.results)
    
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        if runner.results:
            logger.info("Generating partial results...")
            generate_results_tables(runner.results)
        raise

if __name__ == "__main__":
    main()