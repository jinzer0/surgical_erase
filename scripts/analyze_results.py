import optuna
import argparse
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, default="surgical_erase_multi_opt_v19")
    parser.add_argument("--storage", type=str, default="sqlite:///db.sqlite3")
    args = parser.parse_args()
    
    print(f"Loading study '{args.study_name}' from {args.storage}...")
    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    print("\nPareto Front Solutions:")
    best_trials = study.best_trials
    
    candidates = []
    for t in best_trials:
        nudenet_val = t.values[0]
        scaling_val = t.values[1]
        print(f"  Trial {t.number}: NudeNet={nudenet_val}, Scaling={scaling_val:.4f}")
        print(f"    Params: {t.params}")
        candidates.append((nudenet_val, scaling_val, t))
        
    if not candidates:
        print("No completed trials yet.")
        return

    # Sort by NudeNet count (primary) then Scaling (secondary)
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    best_candidate = candidates[0]
    best_nudenet, best_scaling, best_trial = best_candidate
    
    print(f"\nBest Candidate (Lowest NudeNet):")
    print(f"  Trial {best_trial.number}")
    print(f"  NudeNet: {best_nudenet}")
    print(f"  Scaling: {best_scaling:.4f}%")
    print(f"  Params: {json.dumps(best_trial.params, indent=4)}")

if __name__ == "__main__":
    main()
