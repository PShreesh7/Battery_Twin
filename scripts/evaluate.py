import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nasa_loader import NASADataLoader
from src.data.calce_loader import CALCEDataLoader
from src.features.extractor import FeatureExtractor
from src.data.unified_dataset import BatterySequenceDataset, prepare_temporal_splits
from src.physics.baseline import PhysicsOnlyBaseline
from src.models.ai_lstm import AIBatteryLSTM
from src.models.pinn_lstm import PINNBatteryLSTM
from src.preprocessing.scaler import FeatureScaler
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.physics_checks import PhysicalPlausibilityEngine
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")

def run_benchmark():
    logger.info("Executing Controlled Three-Model Benchmark...")
    nasa_loader = NASADataLoader()
    calce_loader = CALCEDataLoader()
    extractor = FeatureExtractor()
    plausibility = PhysicalPlausibilityEngine()
    
    cells_to_bench = [
        ("nasa", "B0005", nasa_loader.load_cell("B0005")),
        ("nasa", "B0006", nasa_loader.load_cell("B0006")),
        ("calce", "CS2_35", calce_loader.load_cell("CS2_35")),
        ("calce", "CS2_36", calce_loader.load_cell("CS2_36"))
    ]
    
    comparison_rows = []
    
    for dataset_name, cell_id, raw_df in cells_to_bench:
        df_feat = extractor.extract_features(raw_df)
        feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
        seq_length = 5
        
        splits = prepare_temporal_splits(df_feat, feature_cols, target_col="capacity", seq_length=seq_length)
        scaler = splits["scaler"]
        test_data = splits["test"]
        raw_test_df = splits["raw_splits"][2]
        raw_train_df = splits["raw_splits"][0]
        
        test_ds = BatterySequenceDataset(*test_data)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        y_true = raw_test_df["capacity"].values[seq_length - 1:]
        test_cycles = raw_test_df["cycle"].values[seq_length - 1:]
        
        # 1. Physics-Only Model
        phys_model = PhysicsOnlyBaseline.load(f"artifacts/physics/{dataset_name}_{cell_id}_physics.json")
        y_pred_phys = phys_model.predict(test_cycles)
        m_phys = compute_regression_metrics(y_true, y_pred_phys)
        plaus_phys = plausibility.evaluate_plausibility(y_pred_phys, cell_id)
        
        comparison_rows.append({
            "Dataset": dataset_name.upper(),
            "Cell": cell_id,
            "Model": "Physics-Only",
            "RMSE": round(m_phys["rmse"], 4),
            "MAE": round(m_phys["mae"], 4),
            "MAPE (%)": round(m_phys["mape"], 2),
            "R2": round(m_phys["r2"], 4),
            "Plausibility Score": plaus_phys["plausibility_score"],
            "Monotonicity Violations": plaus_phys["monotonicity_violations"],
            "Status": plaus_phys["status"]
        })
        
        # 2. AI-Only Model
        ai_model = AIBatteryLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2)
        ai_model.load_state_dict(torch.load(f"artifacts/ai/{dataset_name}_{cell_id}/best_model.pt", weights_only=True))
        ai_model.eval()
        
        ai_preds_scaled = []
        with torch.no_grad():
            for X_b, _, _, _ in test_loader:
                ai_preds_scaled.extend(ai_model(X_b).squeeze(-1).numpy())
        y_pred_ai = scaler.inverse_transform_target(np.array(ai_preds_scaled), "capacity")
        m_ai = compute_regression_metrics(y_true, y_pred_ai)
        plaus_ai = plausibility.evaluate_plausibility(y_pred_ai, cell_id)
        
        comparison_rows.append({
            "Dataset": dataset_name.upper(),
            "Cell": cell_id,
            "Model": "AI-Only (LSTM)",
            "RMSE": round(m_ai["rmse"], 4),
            "MAE": round(m_ai["mae"], 4),
            "MAPE (%)": round(m_ai["mape"], 2),
            "R2": round(m_ai["r2"], 4),
            "Plausibility Score": plaus_ai["plausibility_score"],
            "Monotonicity Violations": plaus_ai["monotonicity_violations"],
            "Status": plaus_ai["status"]
        })
        
        # 3. Hybrid PINN Model
        with open(f"artifacts/pinn/{dataset_name}_{cell_id}/pinn_config.json", "r") as f:
            pinn_cfg = json.load(f)
            
        pinn_model = PINNBatteryLSTM(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            equation_name=pinn_cfg["equation_name"],
            physics_params=pinn_cfg["physics_params"],
            lambda_physics=pinn_cfg["lambda_physics"],
            gamma_monotonicity=pinn_cfg["gamma_monotonicity"]
        )
        pinn_model.load_state_dict(torch.load(f"artifacts/pinn/{dataset_name}_{cell_id}/best_model.pt", weights_only=True))
        pinn_model.eval()
        
        pinn_preds_scaled = []
        with torch.no_grad():
            for X_b, _, _, _ in test_loader:
                pinn_preds_scaled.extend(pinn_model(X_b).squeeze(-1).numpy())
        y_pred_pinn = scaler.inverse_transform_target(np.array(pinn_preds_scaled), "capacity")
        m_pinn = compute_regression_metrics(y_true, y_pred_pinn)
        plaus_pinn = plausibility.evaluate_plausibility(y_pred_pinn, cell_id)
        
        comparison_rows.append({
            "Dataset": dataset_name.upper(),
            "Cell": cell_id,
            "Model": "Hybrid PINN",
            "RMSE": round(m_pinn["rmse"], 4),
            "MAE": round(m_pinn["mae"], 4),
            "MAPE (%)": round(m_pinn["mape"], 2),
            "R2": round(m_pinn["r2"], 4),
            "Plausibility Score": plaus_pinn["plausibility_score"],
            "Monotonicity Violations": plaus_pinn["monotonicity_violations"],
            "Status": plaus_pinn["status"]
        })
        
    df_comp = pd.DataFrame(comparison_rows)
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(out_dir / "model_comparison.csv", index=False)
    df_comp.to_csv(out_dir / "table_model_comparison.csv", index=False)
    
    # Generate Benchmark Plot for NASA B0005 & CALCE CS2_35
    fig_dir = Path("results/figures/models")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 6), dpi=300)
    for idx, (dname, cid) in enumerate([("NASA", "B0005"), ("CALCE", "CS2_35")]):
        sub = df_comp[(df_comp["Dataset"] == dname) & (df_comp["Cell"] == cid)]
        plt.subplot(1, 2, idx + 1)
        models = sub["Model"].values
        rmses = sub["RMSE"].values
        colors = ["#2b5c8f", "#d95f02", "#1b9e77"]
        bars = plt.bar(models, rmses, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
        plt.ylabel("Test RMSE (Ah)", fontsize=11)
        plt.title(f"{dname} {cid} - Model Comparison", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle=":", alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig(fig_dir / "three_model_comparison.png")
    plt.close()
    
    logger.info("Three-model benchmark report generated at results/metrics/model_comparison.csv")

if __name__ == "__main__":
    run_benchmark()
