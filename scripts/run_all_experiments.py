import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("run_all_experiments")

STAGES = [
    ("Data Download", "scripts/download_data.py"),
    ("Data Validation", "scripts/validate_data.py"),
    ("Preprocessing & Feature Engineering", "scripts/preprocess.py"),
    ("Physics Baseline Training", "scripts/train_physics.py"),
    ("AI Baseline Training", "scripts/train_ai.py"),
    ("Hybrid PINN Training", "scripts/train_pinn.py"),
    ("Three-Model Benchmark Evaluation", "scripts/evaluate.py"),
    ("Long-Horizon Forecasting Experiment", "scripts/forecast_experiment.py"),
    ("Low-Data Efficiency Experiment", "scripts/low_data_experiment.py"),
    ("Physics Weight Ablation Study", "scripts/ablation_study.py"),
    ("Uncertainty Quantification Analysis", "scripts/uncertainty_experiment.py")
]

def main():
    logger.info("================================================================")
    logger.info("STARTING COMPLETE END-TO-END BATTERY TWIN REPRODUCIBILITY RUN")
    logger.info("================================================================")
    
    python_exec = sys.executable
    
    for idx, (name, script) in enumerate(STAGES, 1):
        logger.info(f"[{idx}/{len(STAGES)}] Executing: {name} ({script})...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        res = subprocess.run([python_exec, script], capture_output=True, text=True, env=env)
        if res.returncode != 0:
            logger.error(f"Stage '{name}' failed with error:\n{res.stderr}")
            sys.exit(1)
        else:
            logger.info(f"Stage '{name}' completed successfully.")
            
    logger.info("================================================================")
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! REPRODUCIBILITY VERIFIED.")
    logger.info("================================================================")

if __name__ == "__main__":
    main()
