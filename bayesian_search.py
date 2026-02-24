import optuna
import subprocess
import os
import shutil
import re
import argparse
import sys
import json
import pandas as pd
import functools
OPTUNA_STORAGE="postgresql+psycopg2://optuna:0921@127.0.0.1:5432/optuna"

def get_prompts_from_indices(csv_path, idx_file=None, num_prompts=315):
    """
    Read prompts from csv. If idx_file provided, use it. Else use first N.
    """
    df = pd.read_csv(csv_path)
    
    if idx_file and os.path.exists(idx_file):
        # Read indices
        with open(idx_file, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip().isdigit()]
        selected_indices = indices[:num_prompts]
        if "prompt" in df.columns:
            prompts = df.loc[selected_indices, "prompt"].tolist()
        else:
            prompts = df.iloc[selected_indices, 1].tolist()

        if "evaluation_seed" in df.columns:
            evaluation_seeds = df.loc[selected_indices, "evaluation_seed"].tolist()
        else:
            evaluation_seeds = [None] * len(selected_indices)
        
    else:
        print(f"Index file {idx_file} not found or not provided. Using first {num_prompts} prompts.")
        if "prompt" in df.columns:
            prompts = df["prompt"].tolist()[:num_prompts]
        else:
            prompts = df.iloc[:num_prompts, 1].tolist()
        
        evaluation_seeds = df["evaluation_seed"].tolist()[:num_prompts] if "evaluation_seed" in df.columns else [None]*len(prompts)
            
    return prompts, evaluation_seeds

def objective(trial, prompts, evaluation_seeds):
    # 1. Hyperparameters to tune (Analysis-based Search Space v9 - Delayed + Soft Clamp)
    # tau: Clamp Limit (Magitude of projection)
    # Analysis: Naked (~2.6), Neutral Person (~1.0), Dog (~0.8)
    # Range: 1.0 (Neutral) ~ 3.0 (Allow strong nudity, just cap extreme outlier)
    tau = trial.suggest_float("tau", 0.7, 1.5)
    
    # We define them here properly
    T = trial.suggest_float("T", 0.1, 0.25)
    alpha_max = trial.suggest_float("alpha_max", 0.4, 0.8)
    top_m = trial.suggest_int("top_m", 2, 10)
    eta = trial.suggest_float("eta", 0.05, 0.15)
    ridge = trial.suggest_float("ridge", 20.0, 100.0)
    # steering_scale: Orthogonal Injection uses full vector magnitude.
    # Scaled down to prevent noise in eos_delta.
    steering_scale = trial.suggest_float("steering_scale", 0.5, 1.5)

    # start_step: Delayed Steering (0 ~ 30)
    start_step = trial.suggest_int("start_step", 3, 15)
    
    # end_step: Range-Bounded Steering (25 ~ 50)
    # Allows early stopping of intervention to recover details.
    end_step = trial.suggest_int("end_step", 25, 50)

    # 2. Setup trial-specific paths
    import uuid
    run_id = f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"
    output_dir = f"outputs/optimization_v2/{run_id}"
    temp_csv   = f"{output_dir}/trial_{trial.number}_prompts.csv"
    
    # 3. Construct inference command
    python_executable = sys.executable 
    
    # Convert prompts to list of strings for command line (might be too long?)
    # Better to create a temporary csv for this trial
    os.makedirs(output_dir, exist_ok=True)
    
    # Use global prompts list
    df_prompts = pd.DataFrame({"prompt": prompts, "evaluation_seed": evaluation_seeds})
    df_prompts.to_csv(temp_csv, index=False)
    
    cmd_inference = [
        python_executable, "inference.py",
        "--csvfile", os.path.abspath(temp_csv),
        "--output_dir", output_dir,
        "--fp16",
        "--align_mode", "eos_delta",
        "--modifiers_json", "data/modifiers_v3.json",
        "--num_pairs", "600",
        "--tau", str(tau),
        "--T", str(T),
        "--alpha_max", str(alpha_max),
        "--top_m", str(top_m),
        "--eta", str(eta),
        "--ridge", str(ridge),
        "--steering_scale", str(steering_scale),
        "--start_step", str(start_step),
        "--end_step", str(end_step)
        # num_prompts is implicitly handled by the csv content
    ]

    print(f"\n[Trial {run_id}] Running Inference...")
    try:
        subprocess.run(cmd_inference, check=True, stdout=subprocess.DEVNULL) # Keep stdout silent but let stderr show
    except subprocess.CalledProcessError as e:
        print(f"[Trial {run_id}] Inference Failed: {e}")
        raise RuntimeError(f"Inference failed for trial {run_id}")

    # 4. Construct evaluation command
    cmd_eval = [
        python_executable, "evaluate_by_nudenet.py",
        "--image_dir", output_dir
    ]

    print(f"[Trial {run_id}] Running Evaluation...")
    try:
        subprocess.run(cmd_eval, check=True, stdout=subprocess.DEVNULL) # Keep stdout silent but let stderr show
    except subprocess.CalledProcessError as e:
        print(f"[Trial {run_id}] Evaluation Failed: {e}")
        raise RuntimeError(f"Evaluation failed for trial {run_id}")

    # 5. Parse results
    # NudeNet Count
    log_file_path = f"{output_dir}_nudenet_result.log"
    json_file_path = f"{output_dir}_nudenet_detect.json"
    nudenet_count = float('inf')
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            content = f.read()
            match = re.search(r"Total Number of NudeNet Detected:\s*(\d+)", content)
            if match:
                nudenet_count = int(match.group(1))
    
    # Avg Scaling %
    inference_log_path = os.path.join(output_dir, "inference_log.csv")
    avg_scaling = float('inf')
    if os.path.exists(inference_log_path):
        try:
            df_log = pd.read_csv(inference_log_path)
            if "avg_scaling_pct" in df_log.columns:
                avg_scaling = df_log["avg_scaling_pct"].mean()
        except Exception:
            pass

    # 6. Cleanup
    if os.path.exists(output_dir) and not (100 < nudenet_count < 200):
        shutil.rmtree(output_dir)
    if os.path.exists(log_file_path) and not (100 < nudenet_count < 200):
        os.remove(log_file_path)
    if os.path.exists(temp_csv) and not (100 < nudenet_count < 200):
        os.remove(temp_csv)
    if os.path.exists(json_file_path) and not (100 < nudenet_count < 200):
        os.remove(json_file_path)
        

    print(f"[Trial {run_id}] Result: NudeNet={nudenet_count}, Scaling%={avg_scaling:.4f}")
    return nudenet_count, avg_scaling

class MaxTrialsCallback:
    def __init__(self, max_trials):
        self.max_trials = max_trials

    def __call__(self, study, trial):
        n_complete = len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE]])
        if n_complete >= self.max_trials:
            study.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=200, help="Number of trials for optimization")
    parser.add_argument("--num_prompts", type=int, default=315, help="Number of prompts to use for evaluation")
    parser.add_argument("--storage", type=str, default=OPTUNA_STORAGE, help="Optuna storage URL")
    parser.add_argument("--study_name", type=str, default="surgical_erase_multi_opt_v13", help="Optuna study name")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of concurrent trials per GPU (approx 5GB VRAM per trial)")
    
    shutil.rmtree("outputs/optimization_v2/", ignore_errors=True)
    os.makedirs("outputs/optimization_v2/", exist_ok=True)

    global ARGS
    ARGS = parser.parse_args()
    
    # Prepare prompts
    global TARGET_PROMPTS, EVALUATION_SEEDS
    TARGET_PROMPTS, EVALUATION_SEEDS = get_prompts_from_indices(
        "./unsafe_prompt4703.csv", 
        "./nudity_idx.txt", 
        num_prompts=ARGS.num_prompts
    )
    print(f"Loaded {len(TARGET_PROMPTS)} prompts")

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        seed=42
    )

    study = optuna.create_study(
        directions=["minimize", "minimize"], # Multi-objective
        storage=ARGS.storage,
        study_name=ARGS.study_name,
        load_if_exists=True,
        sampler=sampler
    )
    
    print(f"Starting Multi-Objective optimization with target total {ARGS.n_trials} trials...")
    
    # Calculate how many trials are already done to adjust logging/expectations
    n_current_trials = len(study.trials)
    print(f"Current study has {n_current_trials} trials.")

    if n_current_trials >= ARGS.n_trials:
        print(f"Study already has {n_current_trials} trials, which meets or exceeds the target of {ARGS.n_trials}. Exiting.")
    else:
        study_objective = functools.partial(objective, prompts=TARGET_PROMPTS, evaluation_seeds=EVALUATION_SEEDS)
        
        # We use a callback to stop the study when the GLOBAL trial count reaches n_trials
        # This allows multiple processes to coordinate and stop at the correct total number.
        study.optimize(
            study_objective, 
            n_trials=ARGS.n_trials, # Make each process try to reach the limit individually (effectively overridden by callback for global count)
            n_jobs=ARGS.n_jobs, 
            callbacks=[MaxTrialsCallback(ARGS.n_trials)],
            catch=(RuntimeError,)
        )

    print("\nOptimization Completed (or Stopped by Callback)!")
    
    # Pareto Analysis
    print("\nPareto Front Solutions:")
    best_trials = study.best_trials
    
    candidates = []
    for t in best_trials:
        nudenet_val = t.values[0]
        scaling_val = t.values[1]
        print(f"  Trial {t.number}: NudeNet={nudenet_val}, Scaling={scaling_val:.4f}, Params={t.params}")
        candidates.append((nudenet_val, scaling_val, t))
        
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    if candidates:
        best_candidate = candidates[0]
        best_nudenet, best_scaling, best_trial = best_candidate
        
        print(f"\nSelected Best Solution (Safe & Minimal):")
        print(f"  Trial {best_trial.number}")
        print(f"  NudeNet Count: {best_nudenet}")
        print(f"  Avg Scaling %: {best_scaling:.4f}")
        print(f"  Params: {best_trial.params}")

        # Save results to result.md
        result_md_path = "result.md"
        with open(result_md_path, "a") as f:
            f.write("\n\n## Multi-Objective Bayesian Optimization Result (Review Request)\n")
            f.write(f"Run configuration : n_trials={ARGS.n_trials}, num_prompts={len(TARGET_PROMPTS)} (from nudity_idx.txt)\n")
            f.write(f"Selected Best Trial: {best_trial.number}\n")
            f.write(f"Best Params: {json.dumps(best_trial.params, indent=4)}\n")
            f.write(f"Metrics: NudeNet={best_nudenet}, Scaling%={best_scaling:.4f}\n")
        
        # --- Verification Run with Visualization ---
        print("\n[Verification] Running final inference with BEST params and VISUALIZATION...")
        verif_output_dir = "outputs/final_verification_" + ARGS.study_name.split("_")[-1]
        if os.path.exists(verif_output_dir):
            shutil.rmtree(verif_output_dir)
        os.makedirs(verif_output_dir, exist_ok=True)
        
        # Save prompts to csv for verification
        verif_csv = "unsafe_prompt315.csv"
        pd.DataFrame({"prompt": TARGET_PROMPTS, "evaluation_seed": EVALUATION_SEEDS}).to_csv(verif_csv, index=False)
        
        bp = best_trial.params
        python_executable = sys.executable
        
        cmd_verif = [
            python_executable, "inference.py",
            "--csvfile", verif_csv,
            "--output_dir", verif_output_dir,
            "--fp16",
            "--align_mode", "eos_delta",
            "--modifiers_json", "data/modifiers_v3.json",
            "--num_pairs", "600",
            "--tau", str(bp["tau"]),
            "--T", str(bp["T"]),
            "--alpha_max", str(bp["alpha_max"]),
            "--top_m", str(bp["top_m"]),
            "--eta", str(bp["eta"]),
            "--ridge", str(bp["ridge"]),
            "--steering_scale", str(bp["steering_scale"]),
            "--start_step", str(bp["start_step"]),
            "--end_step", str(bp["end_step"]),
            "--visualize" # Enable visualization
        ]
        
        try:
            subprocess.run(cmd_verif, check=True)
            print(f"Verification inference completed. Results in {verif_output_dir}")
            
            # Run Evaluation on Verification Output
            cmd_eval_verif = [
                python_executable, "evaluate_by_nudenet.py",
                "--image_dir", verif_output_dir
            ]
            subprocess.run(cmd_eval_verif, check=True)
            
            # Append final eval result to result.md
            log_file_verif = f"{verif_output_dir}_nudenet_result.log"
            if os.path.exists(log_file_verif):
                 with open(log_file_verif, 'r') as log_f:
                    log_content = log_f.read()
                    print(f"\nFinal Verification NudeNet Result:\n{log_content}")
                    with open(result_md_path, "a") as res_f:
                        res_f.write("\n### Final Verification NudeNet Result\n")
                        res_f.write(log_content)
            
        except subprocess.CalledProcessError as e:
            print(f"Verification failed: {e}")

    else:
        print("No valid trials found.")
