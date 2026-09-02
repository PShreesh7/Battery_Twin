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
from src.models.pinn_lstm import PINNBatteryLSTM
from src.training.trainer import BatteryModelTrainer
from src.physics.fitting import select_best_physics_equation
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.physics_checks import PhysicalPlausibilityEngine
from src.utils.logger import setup_logger

logger = setup_logger("ablation_study")

def run_ablation():
    logger.info("Executing Systematic Physics Loss Weight (lambda) Ablation Study...")
    nasa_loader = NASADataLoader()
    extractor = FeatureExtractor()
    plausibility = PhysicalPlausibilityEngine()
    
    df = nasa_loader.load_cell("B0005")
    df_feat = extractor.extract_features(df)
    feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
    
    splits = prepare_temporal_splits(df_feat, feature_cols, target_col="capacity", seq_length=5)
    scaler = splits["scaler"]
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    raw_train_df = splits["raw_splits"][0]
    
    train_loader = DataLoader(BatterySequenceDataset(*train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(BatterySequenceDataset(*val_data), batch_size=16, shuffle=False)
    test_loader = DataLoader(BatterySequenceDataset(*test_data), batch_size=16, shuffle=False)
    
    q0 = float(raw_train_df["capacity"].iloc[0])
    best_eq, phys_params, _ = select_best_physics_equation(
        raw_train_df["cycle"].values,
        raw_train_df["capacity"].values,
        q0_initial=q0
    )
    
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 1.0]
    results = []
    
    for lam in lambdas:
        model = PINNBatteryLSTM(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            equation_name=best_eq,
            physics_params=phys_params,
            lambda_physics=lam
        )
        trainer = BatteryModelTrainer(
            model=model,
            experiment_id=f"EXP_ABLATION_LAM_{str(lam).replace('.', '_')}",
            output_dir=f"artifacts/pinn/ablation_lam_{str(lam).replace('.', '_')}",
            is_pinn=(lam > 0),
            epochs=100,
            patience=25
        )
        trainer.fit(train_loader, val_loader)
        
        model.eval()
        preds_scaled, y_trues_scaled = [], []
        with torch.no_grad():
            for X_b, y_b, _, _ in test_loader:
                preds_scaled.extend(model(X_b).squeeze(-1).numpy())
                y_trues_scaled.extend(y_b.squeeze(-1).numpy())
        y_pred = scaler.inverse_transform_target(np.array(preds_scaled), "capacity")
        y_true = scaler.inverse_transform_target(np.array(y_trues_scaled), "capacity")
        
        m = compute_regression_metrics(y_true, y_pred)
        plaus = plausibility.evaluate_plausibility(y_pred, "B0005")
        
        results.append({
            "Lambda_Physics": lam,
            "Model_Description": "Pure AI (No Physics)" if lam == 0.0 else f"PINN (lambda={lam})",
            "RMSE": round(m["rmse"], 4),
            "MAE": round(m["mae"], 4),
            "MAPE (%)": round(m["mape"], 2),
            "R2": round(m["r2"], 4),
            "Plausibility Score": plaus["plausibility_score"],
            "Monotonicity Violations": plaus["monotonicity_violations"]
        })
        
    df_ab = pd.DataFrame(results)
    out_dir = Path("results/metrics")
    df_ab.to_csv(out_dir / "table_ablation.csv", index=False)
    
    # Plot Ablation Curves
    fig_dir = Path("results/figures/ablation")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot([str(r["Lambda_Physics"]) for r in results], [r["RMSE"] for r in results], marker="o", color="#7570b3", linewidth=2.2)
    plt.xlabel("Physics Loss Weight (Lambda)", fontsize=11)
    plt.ylabel("Test RMSE (Ah)", fontsize=11)
    plt.title("Ablation Study: Test Error vs Physics Loss Weight (Lambda)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_dir / "lambda_ablation.png")
    plt.close()
    
    logger.info("Ablation study completed and saved to results/metrics/table_ablation.csv")

if __name__ == "__main__":
    run_ablation()
