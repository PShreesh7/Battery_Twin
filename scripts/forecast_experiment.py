import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nasa_loader import NASADataLoader
from src.features.extractor import FeatureExtractor
from src.physics.baseline import PhysicsOnlyBaseline
from src.models.ai_lstm import AIBatteryLSTM
from src.models.pinn_lstm import PINNBatteryLSTM
from src.preprocessing.scaler import FeatureScaler
from src.evaluation.metrics import compute_regression_metrics
from src.utils.logger import setup_logger

logger = setup_logger("forecast_experiment")

def run_forecast_experiment():
    logger.info("Executing Long-Horizon Multi-Step Forecasting Benchmark...")
    nasa_loader = NASADataLoader()
    extractor = FeatureExtractor()
    
    df = nasa_loader.load_cell("B0005")
    df_feat = extractor.extract_features(df)
    
    feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
    seq_length = 5
    
    n_train = int(len(df_feat) * 0.60)
    train_df = df_feat.iloc[:n_train]
    test_df = df_feat.iloc[n_train:]
    
    scaler = FeatureScaler.load("artifacts/pinn/nasa_B0005_scaler.json")
    
    # Load models
    phys_model = PhysicsOnlyBaseline.load("artifacts/physics/nasa_B0005_physics.json")
    
    ai_model = AIBatteryLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2)
    ai_model.load_state_dict(torch.load("artifacts/ai/nasa_B0005/best_model.pt", weights_only=True))
    ai_model.eval()
    
    with open("artifacts/pinn/nasa_B0005/pinn_config.json", "r") as f:
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
    pinn_model.load_state_dict(torch.load("artifacts/pinn/nasa_B0005/best_model.pt", weights_only=True))
    pinn_model.eval()
    
    horizons = [10, 30, 50]
    forecast_results = []
    
    start_idx = n_train
    actual_test_cycles = df_feat["cycle"].values[start_idx:]
    actual_test_capacity = df_feat["capacity"].values[start_idx:]
    
    for h in horizons:
        if h > len(actual_test_capacity):
            continue
            
        y_true_h = actual_test_capacity[:h]
        cycles_h = actual_test_cycles[:h]
        
        # 1. Physics Forecast
        y_pred_phys = phys_model.predict(cycles_h)
        m_phys = compute_regression_metrics(y_true_h, y_pred_phys)
        
        # 2. AI Autoregressive Rollout
        # Initialize input sequence from end of train
        curr_seq = scaler.transform(df_feat.iloc[start_idx - seq_length : start_idx])[feature_cols].values
        ai_preds = []
        for step in range(h):
            inp_t = torch.tensor(curr_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = ai_model(inp_t).item()
            pred_cap = scaler.inverse_transform_target(np.array([pred_scaled]), "capacity")[0]
            ai_preds.append(pred_cap)
            
            # Roll forward sequence
            next_feat = curr_seq[-1].copy()
            # Update cycle and lagged capacity
            next_feat[0] += 1.0 / scaler.stds.get("cycle", 1.0)
            next_feat[5] = pred_scaled
            curr_seq = np.vstack([curr_seq[1:], next_feat])
            
        m_ai = compute_regression_metrics(y_true_h, np.array(ai_preds))
        
        # 3. PINN Autoregressive Rollout
        curr_seq_pinn = scaler.transform(df_feat.iloc[start_idx - seq_length : start_idx])[feature_cols].values
        pinn_preds = []
        for step in range(h):
            inp_t = torch.tensor(curr_seq_pinn, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = pinn_model(inp_t).item()
            pred_cap = scaler.inverse_transform_target(np.array([pred_scaled]), "capacity")[0]
            pinn_preds.append(pred_cap)
            
            next_feat = curr_seq_pinn[-1].copy()
            next_feat[0] += 1.0 / scaler.stds.get("cycle", 1.0)
            next_feat[5] = pred_scaled
            curr_seq_pinn = np.vstack([curr_seq_pinn[1:], next_feat])
            
        m_pinn = compute_regression_metrics(y_true_h, np.array(pinn_preds))
        
        forecast_results.extend([
            {"Horizon": f"Short ({h} cycles)" if h == 10 else f"Medium ({h} cycles)" if h == 30 else f"Long ({h} cycles)", "Horizon_Steps": h, "Model": "Physics-Only", "RMSE": round(m_phys["rmse"], 4), "MAE": round(m_phys["mae"], 4), "MAPE (%)": round(m_phys["mape"], 2), "R2": round(m_phys["r2"], 4)},
            {"Horizon": f"Short ({h} cycles)" if h == 10 else f"Medium ({h} cycles)" if h == 30 else f"Long ({h} cycles)", "Horizon_Steps": h, "Model": "AI-Only (LSTM)", "RMSE": round(m_ai["rmse"], 4), "MAE": round(m_ai["mae"], 4), "MAPE (%)": round(m_ai["mape"], 2), "R2": round(m_ai["r2"], 4)},
            {"Horizon": f"Short ({h} cycles)" if h == 10 else f"Medium ({h} cycles)" if h == 30 else f"Long ({h} cycles)", "Horizon_Steps": h, "Model": "Hybrid PINN", "RMSE": round(m_pinn["rmse"], 4), "MAE": round(m_pinn["mae"], 4), "MAPE (%)": round(m_pinn["mape"], 2), "R2": round(m_pinn["r2"], 4)}
        ])
        
    df_fc = pd.DataFrame(forecast_results)
    out_dir = Path("results/metrics")
    df_fc.to_csv(out_dir / "table_forecast_horizon.csv", index=False)
    
    # Plot Long-Horizon Trajectory Comparison (50 cycles)
    fig_dir = Path("results/figures/forecasts")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(11, 6), dpi=300)
    plt.plot(df_feat["cycle"].iloc[:start_idx], df_feat["capacity"].iloc[:start_idx], color="black", label="Training Observed History", linewidth=2.0)
    plt.plot(actual_test_cycles[:50], actual_test_capacity[:50], color="dimgray", linestyle=":", label="Actual Future Ground Truth", linewidth=2.5)
    plt.plot(actual_test_cycles[:50], phys_model.predict(actual_test_cycles[:50]), color="#2b5c8f", linestyle="--", label="Physics-Only Forecast", linewidth=1.8)
    plt.plot(actual_test_cycles[:50], ai_preds[:50], color="#d95f02", linestyle="-.", label="AI-Only (LSTM) Forecast", linewidth=1.8)
    plt.plot(actual_test_cycles[:50], pinn_preds[:50], color="#1b9e77", linewidth=2.2, label="Hybrid PINN Forecast")
    
    plt.axvline(df_feat["cycle"].iloc[start_idx], color="red", linestyle="--", alpha=0.6, label="Forecast Horizon Start")
    plt.xlabel("Cycle Number", fontsize=11)
    plt.ylabel("Capacity (Ah)", fontsize=11)
    plt.title("NASA B0005 Long-Horizon Forecasting Trajectory Benchmark", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_dir / "long_horizon_forecast.png")
    plt.close()
    
    logger.info("Long-horizon forecast benchmark completed and saved to results/metrics/table_forecast_horizon.csv")

if __name__ == "__main__":
    run_forecast_experiment()
