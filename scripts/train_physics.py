import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nasa_loader import NASADataLoader
from src.data.calce_loader import CALCEDataLoader
from src.physics.baseline import PhysicsOnlyBaseline
from src.utils.logger import setup_logger

logger = setup_logger("train_physics")

def evaluate_physics_on_cell(cell_df: pd.DataFrame, cell_id: str, dataset_name: str):
    n_train = int(len(cell_df) * 0.60)
    n_val = int(len(cell_df) * 0.15)
    
    train_df = cell_df.iloc[:n_train]
    val_df = cell_df.iloc[n_train : n_train + n_val]
    test_df = cell_df.iloc[n_train + n_val :]
    
    q0 = float(train_df["capacity"].iloc[0])
    
    # Fit strictly on train
    model = PhysicsOnlyBaseline()
    model.fit(train_df["cycle"].values, train_df["capacity"].values, q0_initial=q0)
    
    # Predict on test
    test_preds = model.predict(test_df["cycle"].values)
    y_test = test_df["capacity"].values
    
    rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    mae = float(mean_absolute_error(y_test, test_preds))
    mape = float(np.mean(np.abs((y_test - test_preds) / y_test)) * 100.0)
    r2 = float(r2_score(y_test, test_preds))
    
    # Save artifact
    art_path = Path(f"artifacts/physics/{dataset_name}_{cell_id}_physics.json")
    model.save(str(art_path))
    
    results = {
        "dataset": dataset_name,
        "cell_id": cell_id,
        "equation": model.equation_name,
        "params": model.params,
        "test_metrics": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
    }
    logger.info(f"Physics Baseline {dataset_name} {cell_id} ({model.equation_name}): Test RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")
    return results

def main():
    logger.info("Training and Evaluating Physics-Only Baseline...")
    nasa_loader = NASADataLoader()
    calce_loader = CALCEDataLoader()
    
    all_results = []
    
    # Benchmark on NASA cells
    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        df = nasa_loader.load_cell(cid)
        res = evaluate_physics_on_cell(df, cid, "nasa")
        all_results.append(res)
        
    # Benchmark on CALCE cells
    for cid in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        df = calce_loader.load_cell(cid)
        res = evaluate_physics_on_cell(df, cid, "calce")
        all_results.append(res)
        
    out_p = Path("results/metrics/physics_baseline_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    logger.info("Physics Baseline benchmarking completed and saved to results/metrics/physics_baseline_results.json")

if __name__ == "__main__":
    main()
