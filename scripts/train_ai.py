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
from src.models.ai_lstm import AIBatteryLSTM
from src.training.trainer import BatteryModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger("train_ai")

def train_and_eval_ai(cell_df: pd.DataFrame, cell_id: str, dataset_name: str, epochs: int = 150):
    extractor = FeatureExtractor()
    df_feat = extractor.extract_features(cell_df)
    
    feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
    seq_length = 5
    splits = prepare_temporal_splits(df_feat, feature_cols, target_col="capacity", seq_length=seq_length)
    
    scaler = splits["scaler"]
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    
    # Save scaler
    scaler_path = f"artifacts/ai/{dataset_name}_{cell_id}_scaler.json"
    scaler.save(scaler_path)
    
    train_ds = BatterySequenceDataset(*train_data)
    val_ds = BatterySequenceDataset(*val_data)
    test_ds = BatterySequenceDataset(*test_data)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = AIBatteryLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2)
    exp_id = f"EXP_{dataset_name.upper()}_{cell_id}_AI"
    out_dir = f"artifacts/ai/{dataset_name}_{cell_id}"
    
    trainer = BatteryModelTrainer(
        model=model,
        experiment_id=exp_id,
        output_dir=out_dir,
        is_pinn=False,
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
            
    # Inverse transform target capacity
    y_preds = scaler.inverse_transform_target(np.array(y_preds_scaled), "capacity")
    y_trues = scaler.inverse_transform_target(np.array(y_trues_scaled), "capacity")
    
    rmse = float(np.sqrt(mean_squared_error(y_trues, y_preds)))
    mae = float(mean_absolute_error(y_trues, y_preds))
    mape = float(np.mean(np.abs((y_trues - y_preds) / y_trues)) * 100.0)
    r2 = float(r2_score(y_trues, y_preds))
    
    results = {
        "dataset": dataset_name,
        "cell_id": cell_id,
        "model_type": "ai_lstm",
        "test_metrics": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
    }
    logger.info(f"AI-LSTM {dataset_name} {cell_id}: Test RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")
    return results

def main():
    logger.info("Training AI-Only (LSTM) Baseline...")
    nasa_loader = NASADataLoader()
    calce_loader = CALCEDataLoader()
    
    all_results = []
    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        df = nasa_loader.load_cell(cid)
        res = train_and_eval_ai(df, cid, "nasa", epochs=120)
        all_results.append(res)
        
    for cid in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        df = calce_loader.load_cell(cid)
        res = train_and_eval_ai(df, cid, "calce", epochs=120)
        all_results.append(res)
        
    out_p = Path("results/metrics/ai_baseline_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    logger.info("AI-Only Baseline training completed and saved to results/metrics/ai_baseline_results.json")

if __name__ == "__main__":
    main()
