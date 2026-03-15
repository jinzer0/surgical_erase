import optuna
import sys

OPTUNA_STORAGE = "postgresql+psycopg2://optuna:0921@127.0.0.1:5433/optuna"
STUDY_NAME = "surgical_erase_multi_opt_v1"

try:
    study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_STORAGE)
    print(f"Study statistics for '{STUDY_NAME}':")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Running: {len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}")
    print(f"  Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
except Exception as e:
    print(f"Error loading study: {e}")
