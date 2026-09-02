import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nasa_loader import NASADataLoader
from src.data.calce_loader import CALCEDataLoader
from src.features.extractor import FeatureExtractor
from src.data.unified_dataset import BatterySequenceDataset, prepare_temporal_splits
from src.physics.fitting import select_best_physics_equation
from src.models.pinn_lstm import PINNBatteryLSTM
from src.training.trainer import BatteryModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger("train_pinn")

def train_and_eval_pinn(
    cell_df: pd.DataFrame,
    cell_id: str,
    dataset_name: str,
    lambda_physics: float = 0.05,
    gamma_monotonicity: float = 0.5,
    epochs: int = 150
):
    extractor = FeatureExtractor()
    df_feat = extractor.extract_features(cell_df)
    
    feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
    seq_length = 5
    splits = prepare_temporal_splits(df_feat, feature_cols, target_col="capacity", seq_length=seq_length)
    
    scaler = splits["scaler"]
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    raw_train_df = splits["raw_splits"][0]
    
    # Fit physics equation on raw train slice to supply parameters to PINN loss
    q0 = float(raw_train_df["capacity"].iloc[0])
    best_eq, phys_params, _ = select_best_physics_equation(
        raw_train_df["cycle"].values,
        raw_train_df["capacity"].values,
        q0_initial=q0
    )
    
    # Save scaler and physics metadata
    scaler_path = f"artifacts/pinn/{dataset_name}_{cell_id}_scaler.json"
    scaler.save(scaler_path)
    
    train_ds = BatterySequenceDataset(*train_data)
    val_ds = BatterySequenceDataset(*val_data)
    test_ds = BatterySequenceDataset(*test_data)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = PINNBatteryLSTM(
        input_size=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        equation_name=best_eq,
        physics_params=phys_params,
        lambda_physics=lambda_physics,
        gamma_monotonicity=gamma_monotonicity
    )
    
    exp_id = f"EXP_{dataset_name.upper()}_{cell_id}_PINN"
    out_dir = f"artifacts/pinn/{dataset_name}_{cell_id}"
    
    trainer = BatteryModelTrainer(
        model=model,
        experiment_id=exp_id,
        output_dir=out_dir,
        is_pinn=True,
        learning_rate=0.001,
        epochs=epochs,
        patience=30
    )
    
    trainer.fit(train_loader, val_loader)
    
    # Test Evaluation
    model.eval()
    y_preds_scaled, y_trues_scaled = [], []
    with torch.no_grad():
        for X_b, y_b, _, _ in test_loader:
            pred = model(X_b)
            y_preds_scaled.extend(pred.squeeze(-1).numpy())
            y_trues_scaled.extend(y_b.squeeze(-1).numpy())
            
    y_preds = scaler.inverse_transform_target(np.array(y_preds_scaled), "capacity")
    y_trues = scaler.inverse_transform_target(np.array(y_trues_scaled), "capacity")
    
    rmse = float(np.sqrt(mean_squared_error(y_trues, y_preds)))
    mae = float(mean_absolute_error(y_trues, y_preds))
    mape = float(np.mean(np.abs((y_trues - y_preds) / y_trues)) * 100.0)
    r2 = float(r2_score(y_trues, y_preds))
    
    # Save complete model metadata
    pinn_meta = {
        "equation_name": best_eq,
        "physics_params": phys_params,
        "lambda_physics": lambda_physics,
        "gamma_monotonicity": gamma_monotonicity,
        "feature_cols": feature_cols,
        "seq_length": seq_length
    }
    with open(Path(out_dir) / "pinn_config.json", "w", encoding="utf-8") as f:
        json.dump(pinn_meta, f, indent=2)
        
    results = {
        "dataset": dataset_name,
        "cell_id": cell_id,
        "model_type": "hybrid_pinn",
        "equation": best_eq,
        "lambda_physics": lambda_physics,
        "test_metrics": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
    }
    logger.info(f"PINN {dataset_name} {cell_id} ({best_eq}, lambda={lambda_physics}): Test RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")
    return results

def main():
    logger.info("Training Hybrid PINN Models...")
    nasa_loader = NASADataLoader()
    calce_loader = CALCEDataLoader()
    
    all_results = []
    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        df = nasa_loader.load_cell(cid)
        res = train_and_eval_pinn(df, cid, "nasa", lambda_physics=0.05, epochs=120)
        all_results.append(res)
        
    for cid in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        df = calce_loader.load_cell(cid)
        res = train_and_eval_pinn(df, cid, "calce", lambda_physics=0.05, epochs=120)
        all_results.append(res)
        
    out_p = Path("results/metrics/pinn_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    logger.info("PINN training completed and saved to results/metrics/pinn_results.json")

if __name__ == "__main__":
    main()
