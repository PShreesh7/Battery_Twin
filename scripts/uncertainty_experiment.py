import sys
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
from src.preprocessing.scaler import FeatureScaler
from src.uncertainty.mc_dropout import predict_mc_dropout
from src.uncertainty.calibration import evaluate_uncertainty_calibration
from src.utils.logger import setup_logger

logger = setup_logger("uncertainty_experiment")

def run_uncertainty():
    logger.info("Executing Uncertainty Quantification & Calibration Analysis...")
    nasa_loader = NASADataLoader()
    extractor = FeatureExtractor()
    
    df = nasa_loader.load_cell("B0005")
    df_feat = extractor.extract_features(df)
    feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
    
    splits = prepare_temporal_splits(df_feat, feature_cols, target_col="capacity", seq_length=5)
    scaler = splits["scaler"]
    test_data = splits["test"]
    raw_test_df = splits["raw_splits"][2]
    
    test_ds = BatterySequenceDataset(*test_data)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    
    # Load trained PINN
    pinn_model = PINNBatteryLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2)
    pinn_model.load_state_dict(torch.load("artifacts/pinn/nasa_B0005/best_model.pt", weights_only=True))
    
    for X_b, y_b, cycles_b, _ in test_loader:
        # Run MC Dropout with 100 samples
        mean_s, std_s, low_s, high_s = predict_mc_dropout(pinn_model, X_b, n_samples=100, confidence_level=0.90)
        
    y_mean = scaler.inverse_transform_target(mean_s, "capacity")
    y_low = scaler.inverse_transform_target(low_s, "capacity")
    y_high = scaler.inverse_transform_target(high_s, "capacity")
    y_true = scaler.inverse_transform_target(y_b.squeeze(-1).numpy(), "capacity")
    test_cycles = cycles_b.squeeze(-1).numpy()
    
    # Calibration metrics
    calib_90 = evaluate_uncertainty_calibration(y_true, y_low, y_high, nominal_confidence=0.90)
    
    # Also evaluate 95% interval
    _, _, low_95_s, high_95_s = predict_mc_dropout(pinn_model, X_b, n_samples=100, confidence_level=0.95)
    y_low_95 = scaler.inverse_transform_target(low_95_s, "capacity")
    y_high_95 = scaler.inverse_transform_target(high_95_s, "capacity")
    calib_95 = evaluate_uncertainty_calibration(y_true, y_low_95, y_high_95, nominal_confidence=0.95)
    
    uq_table = pd.DataFrame([
        {"Nominal Confidence": "90%", "PICP Empirical Coverage (%)": calib_90["picp_empirical_coverage"], "MPIW Width (Ah)": calib_90["mpiw_mean_width"], "Coverage Error (%)": calib_90["coverage_error"], "Well Calibrated": calib_90["is_well_calibrated"]},
        {"Nominal Confidence": "95%", "PICP Empirical Coverage (%)": calib_95["picp_empirical_coverage"], "MPIW Width (Ah)": calib_95["mpiw_mean_width"], "Coverage Error (%)": calib_95["coverage_error"], "Well Calibrated": calib_95["is_well_calibrated"]}
    ])
    out_dir = Path("results/metrics")
    uq_table.to_csv(out_dir / "table_uncertainty.csv", index=False)
    
    # Plot Calibration and Confidence Bands
    fig_dir = Path("results/figures/uncertainty")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(test_cycles, y_true, "k-o", label="Actual Test Measurements", markersize=4, linewidth=1.5)
    plt.plot(test_cycles, y_mean, color="#1b9e77", label="PINN Predictive Mean", linewidth=2.0)
    plt.fill_between(test_cycles, y_low, y_high, color="#1b9e77", alpha=0.3, label="90% Prediction Interval")
    plt.fill_between(test_cycles, y_low_95, y_high_95, color="#1b9e77", alpha=0.15, label="95% Prediction Interval")
    
    plt.xlabel("Cycle Number", fontsize=11)
    plt.ylabel("Capacity (Ah)", fontsize=11)
    plt.title(f"PINN Monte Carlo Dropout Uncertainty Bounds (PICP 90%: {calib_90['picp_empirical_coverage']}%)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curves.png")
    plt.close()
    
    logger.info("Uncertainty quantification experiment completed and saved to results/metrics/table_uncertainty.csv")

if __name__ == "__main__":
    run_uncertainty()
