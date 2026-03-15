import optuna
import pandas as pd
import sys

# Configuration from bayesian_search.py
STORAGE = "postgresql+psycopg2://optuna:0921@127.0.0.1:5432/optuna"
STUDY_NAME = "surgical_erase_multi_opt_v15"

def analyze():
    print(f"Loading study '{STUDY_NAME}' from {STORAGE}...")
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except KeyError:
        print(f"Study '{STUDY_NAME}' not found. Please check the name.")
        return
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    print(f"\nStudy Statistics:")
    print(f"  Total Trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Completed Trials: {len(completed_trials)}")
    
    if not completed_trials:
        print("No completed trials to analyze.")
        return

    # Pareto Front
    print(f"\n[Pareto Front Solutions] (Trade-off between NudeNet & Scaling)")
    best_trials = study.best_trials
    # Sort for display: primary by NudeNet (val[0]), secondary by Scaling (val[1])
    best_trials.sort(key=lambda t: (t.values[0], t.values[1]))
    
    for t in best_trials:
        nudenet = t.values[0]
        scaling = t.values[1]
        print(f"  Trial {t.number:>3}: NudeNet={nudenet:<3} | Scaling={scaling:>6.2f}% | Params={t.params}")

    # Top 10 by NudeNet Count (Primary Metric)
    df = study.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    
    # Rename columns for readability if needed, usually they are "values_0", "values_1", "params_..."
    mapping = {"values_0": "NudeNet", "values_1": "Scaling%"}
    df = df.rename(columns=mapping)
    
    # Sort
    df_sorted = df.sort_values(by=["NudeNet", "Scaling%"])
    
    print(f"\n[Top 10 Trials by NudeNet Count]")
    cols_to_show = ["number", "NudeNet", "Scaling%"] + [c for c in df.columns if c.startswith("params_")]
    print(df_sorted[cols_to_show].head(10).to_string(index=False))

    # Parameter Importance (if enough trials)
    if len(completed_trials) > 10:
        try:
            print(f"\n[Parameter Importance for NudeNet Count]")
            importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
            for param, score in importance.items():
                print(f"  {param:<15}: {score:.4f}")
        except Exception as e:
            print(f"Could not calculate importance: {e}")

if __name__ == "__main__":
    analyze()
