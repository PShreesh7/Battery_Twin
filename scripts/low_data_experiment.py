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
from src.features.extractor import FeatureExtractor
from src.data.unified_dataset import BatterySequenceDataset, prepare_temporal_splits
from src.models.ai_lstm import AIBatteryLSTM
from src.models.pinn_lstm import PINNBatteryLSTM
from src.training.trainer import BatteryModelTrainer
from src.physics.fitting import select_best_physics_equation
from src.evaluation.metrics import compute_regression_metrics
from src.utils.logger import setup_logger

logger = setup_logger("low_data_experiment")

def run_low_data_benchmark():
    logger.info("Running Low-Data Sample Efficiency Benchmark (20%, 40%, 60%, 80%)...")
    nasa_loader = NASADataLoader()
    extractor = FeatureExtractor()
    
    df = nasa_loader.load_cell("B0005")
    df_feat = extractor.extract_features(df)
    feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
    
    fractions = [0.20, 0.40, 0.60, 0.80]
    results = []
    
    for frac in fractions:
        splits = prepare_temporal_splits(
            df_feat,
            feature_cols,
            target_col="capacity",
            seq_length=5,
            low_data_fraction=frac
        )
        scaler = splits["scaler"]
        train_data = splits["train"]
        val_data = splits["val"]
        test_data = splits["test"]
        raw_train_df = splits["raw_splits"][0]
        
        train_loader = DataLoader(BatterySequenceDataset(*train_data), batch_size=8, shuffle=True)
        val_loader = DataLoader(BatterySequenceDataset(*val_data), batch_size=8, shuffle=False)
        test_loader = DataLoader(BatterySequenceDataset(*test_data), batch_size=8, shuffle=False)
        
        # 1. AI Model on reduced data
        ai_model = AIBatteryLSTM(input_size=len(feature_cols), hidden_size=32, num_layers=2, dropout=0.2)
        ai_trainer = BatteryModelTrainer(
            model=ai_model,
            experiment_id=f"EXP_LOWDATA_AI_{int(frac*100)}",
            output_dir=f"artifacts/ai/low_data_{int(frac*100)}",
            is_pinn=False,
            epochs=100,
            patience=25
        )
        ai_trainer.fit(train_loader, val_loader)
        
        ai_model.eval()
        ai_preds_scaled, y_trues_scaled = [], []
        with torch.no_grad():
            for X_b, y_b, _, _ in test_loader:
                ai_preds_scaled.extend(ai_model(X_b).squeeze(-1).numpy())
                y_trues_scaled.extend(y_b.squeeze(-1).numpy())
        y_pred_ai = scaler.inverse_transform_target(np.array(ai_preds_scaled), "capacity")
        y_true = scaler.inverse_transform_target(np.array(y_trues_scaled), "capacity")
        m_ai = compute_regression_metrics(y_true, y_pred_ai)
        
        # 2. PINN Model on reduced data
        q0 = float(raw_train_df["capacity"].iloc[0])
        best_eq, phys_params, _ = select_best_physics_equation(
            raw_train_df["cycle"].values,
            raw_train_df["capacity"].values,
            q0_initial=q0
        )
        pinn_model = PINNBatteryLSTM(
            input_size=len(feature_cols),
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            equation_name=best_eq,
            physics_params=phys_params,
            lambda_physics=0.05
        )
        pinn_trainer = BatteryModelTrainer(
            model=pinn_model,
            experiment_id=f"EXP_LOWDATA_PINN_{int(frac*100)}",
            output_dir=f"artifacts/pinn/low_data_{int(frac*100)}",
            is_pinn=True,
            epochs=100,
            patience=25
        )
        pinn_trainer.fit(train_loader, val_loader)
        
        pinn_model.eval()
        pinn_preds_scaled = []
        with torch.no_grad():
            for X_b, _, _, _ in test_loader:
                pinn_preds_scaled.extend(pinn_model(X_b).squeeze(-1).numpy())
        y_pred_pinn = scaler.inverse_transform_target(np.array(pinn_preds_scaled), "capacity")
        m_pinn = compute_regression_metrics(y_true, y_pred_pinn)
        
        results.extend([
            {"Training_Fraction": f"{int(frac*100)}%", "Fraction_Value": frac, "Model": "AI-Only (LSTM)", "RMSE": round(m_ai["rmse"], 4), "MAE": round(m_ai["mae"], 4), "R2": round(m_ai["r2"], 4)},
            {"Training_Fraction": f"{int(frac*100)}%", "Fraction_Value": frac, "Model": "Hybrid PINN", "RMSE": round(m_pinn["rmse"], 4), "MAE": round(m_pinn["mae"], 4), "R2": round(m_pinn["r2"], 4)}
        ])
        
    df_res = pd.DataFrame(results)
    out_dir = Path("results/metrics")
    df_res.to_csv(out_dir / "data_efficiency_comparison.csv", index=False)
    df_res.to_csv(out_dir / "table_low_data.csv", index=False)
    
    # Plot Data Efficiency Curves
    fig_dir = Path("results/figures/ablation")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 5), dpi=300)
    ai_sub = df_res[df_res["Model"] == "AI-Only (LSTM)"]
    pinn_sub = df_res[df_res["Model"] == "Hybrid PINN"]
    
    plt.plot(ai_sub["Fraction_Value"] * 100, ai_sub["RMSE"], marker="o", color="#d95f02", linewidth=2.0, label="AI-Only Baseline")
    plt.plot(pinn_sub["Fraction_Value"] * 100, pinn_sub["RMSE"], marker="s", color="#1b9e77", linewidth=2.2, label="Hybrid PINN")
    
    plt.xlabel("Available Training History Fraction (%)", fontsize=11)
    plt.ylabel("Test RMSE (Ah)", fontsize=11)
    plt.title("Sample Efficiency: Test Error vs Training Data Availability (H4)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "data_efficiency.png")
    plt.close()
    
    logger.info("Low-data efficiency experiment completed and saved to results/metrics/data_efficiency_comparison.csv")

if __name__ == "__main__":
    run_low_data_benchmark()
