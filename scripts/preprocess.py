import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nasa_loader import NASADataLoader
from src.data.calce_loader import CALCEDataLoader
from src.features.extractor import FeatureExtractor
from src.utils.logger import setup_logger

logger = setup_logger("preprocess")

def main():
    logger.info("Starting preprocessing & feature engineering...")
    nasa_loader = NASADataLoader()
    calce_loader = CALCEDataLoader()
    extractor = FeatureExtractor()
    
    nasa_df = nasa_loader.load_all()
    calce_df = calce_loader.load_all()
    
    nasa_features = extractor.extract_features(nasa_df)
    calce_features = extractor.extract_features(calce_df)
    
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    nasa_features.to_csv(proc_dir / "nasa_processed.csv", index=False)
    calce_features.to_csv(proc_dir / "calce_processed.csv", index=False)
    
    logger.info("Saved processed datasets to data/processed/")
    
    # Generate EDA Figures
    eda_dir = Path("results/figures/eda")
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Capacity vs Cycle for NASA & CALCE
    plt.figure(figsize=(12, 5), dpi=300)
    plt.subplot(1, 2, 1)
    for cid, g in nasa_features.groupby("cell_id"):
        plt.plot(g["cycle"], g["capacity"], label=f"Cell {cid}", linewidth=1.8)
    plt.axhline(1.856 * 0.8, color="red", linestyle="--", alpha=0.7, label="80% EOL Threshold")
    plt.xlabel("Cycle Number", fontsize=11)
    plt.ylabel("Capacity (Ah)", fontsize=11)
    plt.title("NASA PCoE Battery Degradation Profiles", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    
    plt.subplot(1, 2, 2)
    for cid, g in calce_features.groupby("cell_id"):
        plt.plot(g["cycle"], g["capacity"], label=f"Cell {cid}", linewidth=1.8)
    plt.axhline(1.10 * 0.8, color="red", linestyle="--", alpha=0.7, label="80% EOL Threshold")
    plt.xlabel("Cycle Number", fontsize=11)
    plt.ylabel("Capacity (Ah)", fontsize=11)
    plt.title("CALCE CS2 Battery Degradation Profiles", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(eda_dir / "eda_capacity_degradation.png")
    plt.close()
    
    # 2. Voltage, Current, Temperature EDA for NASA B0005
    b0005 = nasa_features[nasa_features["cell_id"] == "B0005"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=300, sharex=True)
    
    axes[0].plot(b0005["cycle"], b0005["voltage_mean"], color="navy", label="Mean Voltage")
    axes[0].fill_between(b0005["cycle"], b0005["voltage_min"], b0005["voltage_max"], color="blue", alpha=0.2, label="Voltage Min-Max Envelope")
    axes[0].set_ylabel("Voltage (V)", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].set_title("NASA B0005 Multi-Signal Evolution across Aging", fontsize=12, fontweight="bold")
    
    axes[1].plot(b0005["cycle"], b0005["temp_mean"], color="darkred", label="Mean Temperature")
    axes[1].fill_between(b0005["cycle"], b0005["temp_min"], b0005["temp_max"], color="red", alpha=0.2, label="Temperature Min-Max Envelope")
    axes[1].set_ylabel("Temperature (?C)", fontsize=10)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    axes[2].plot(b0005["cycle"], b0005["soh"] * 100, color="green", label="State of Health (SOH %)")
    axes[2].axhline(80, color="crimson", linestyle="--", label="80% EOL")
    axes[2].set_ylabel("SOH (%)", fontsize=10)
    axes[2].set_xlabel("Cycle Number", fontsize=11)
    axes[2].legend(loc="lower left")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(eda_dir / "eda_b0005_signals.png")
    plt.close()
    
    logger.info("EDA figures generated in results/figures/eda/")

if __name__ == "__main__":
    main()
