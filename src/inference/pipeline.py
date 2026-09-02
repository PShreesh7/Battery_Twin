import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from src.features.extractor import FeatureExtractor
from src.physics.baseline import PhysicsOnlyBaseline
from src.models.ai_lstm import AIBatteryLSTM
from src.models.pinn_lstm import PINNBatteryLSTM
from src.preprocessing.scaler import FeatureScaler
from src.uncertainty.mc_dropout import predict_mc_dropout
from src.evaluation.physics_checks import PhysicalPlausibilityEngine
from src.utils.logger import setup_logger

logger = setup_logger("inference_pipeline")

class BatteryTwinInferencePipeline:
    """Production-grade, stateless inference service for Battery Digital Twin."""
    
    def __init__(self, artifacts_dir: str = "artifacts", data_dir: str = "data/raw"):
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        self.extractor = FeatureExtractor()
        self.plausibility_engine = PhysicalPlausibilityEngine()
        self._cached_models: Dict[str, Any] = {}
        self._cached_scalers: Dict[str, FeatureScaler] = {}
        
    def get_dataset_for_cell(self, cell_id: str) -> str:
        if cell_id.startswith("B0"):
            return "nasa"
        return "calce"

    def get_initial_capacity(self, cell_id: str) -> float:
        nominals = {
            "B0005": 1.8565, "B0006": 2.0353, "B0007": 1.8911, "B0018": 1.8550,
            "CS2_35": 1.112, "CS2_36": 1.085, "CS2_37": 1.092, "CS2_38": 1.050
        }
        return nominals.get(cell_id, 2.0)

    def load_scaler(self, dataset_name: str, cell_id: str) -> FeatureScaler:
        key = f"{dataset_name}_{cell_id}"
        if key not in self._cached_scalers:
            p = self.artifacts_dir / f"pinn/{dataset_name}_{cell_id}_scaler.json"
            if not p.exists():
                p = self.artifacts_dir / f"ai/{dataset_name}_{cell_id}_scaler.json"
            if not p.exists():
                raise FileNotFoundError(f"Scaler artifact not found for {cell_id} at {p}")
            self._cached_scalers[key] = FeatureScaler.load(str(p))
        return self._cached_scalers[key]

    def load_physics_model(self, dataset_name: str, cell_id: str) -> PhysicsOnlyBaseline:
        key = f"physics_{dataset_name}_{cell_id}"
        if key not in self._cached_models:
            p = self.artifacts_dir / f"physics/{dataset_name}_{cell_id}_physics.json"
            if not p.exists():
                raise FileNotFoundError(f"Physics artifact not found for {cell_id} at {p}")
            self._cached_models[key] = PhysicsOnlyBaseline.load(str(p))
        return self._cached_models[key]

    def load_ai_model(self, dataset_name: str, cell_id: str) -> AIBatteryLSTM:
        key = f"ai_{dataset_name}_{cell_id}"
        if key not in self._cached_models:
            p = self.artifacts_dir / f"ai/{dataset_name}_{cell_id}/best_model.pt"
            if not p.exists():
                raise FileNotFoundError(f"AI model weights not found at {p}")
            model = AIBatteryLSTM(input_size=len(FeatureExtractor.DEFAULT_FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.2)
            model.load_state_dict(torch.load(p, weights_only=True))
            model.eval()
            self._cached_models[key] = model
        return self._cached_models[key]

    def load_pinn_model(self, dataset_name: str, cell_id: str) -> PINNBatteryLSTM:
        key = f"pinn_{dataset_name}_{cell_id}"
        if key not in self._cached_models:
            cfg_p = self.artifacts_dir / f"pinn/{dataset_name}_{cell_id}/pinn_config.json"
            weights_p = self.artifacts_dir / f"pinn/{dataset_name}_{cell_id}/best_model.pt"
            if not weights_p.exists():
                raise FileNotFoundError(f"PINN model weights not found at {weights_p}")
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model = PINNBatteryLSTM(
                input_size=len(FeatureExtractor.DEFAULT_FEATURE_COLS),
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                equation_name=cfg["equation_name"],
                physics_params=cfg["physics_params"],
                lambda_physics=cfg["lambda_physics"],
                gamma_monotonicity=cfg["gamma_monotonicity"]
            )
            model.load_state_dict(torch.load(weights_p, weights_only=True))
            model.eval()
            self._cached_models[key] = model
        return self._cached_models[key]

    def predict_capacity(
        self,
        cell_id: str,
        model_type: str = "pinn",
        feature_sequence: Optional[List[List[float]]] = None,
        cycle: Optional[int] = None
    ) -> float:
        dname = self.get_dataset_for_cell(cell_id)
        if model_type == "physics":
            phys = self.load_physics_model(dname, cell_id)
            c = cycle if cycle is not None else 100
            return float(phys.predict(np.array([c]))[0])
            
        scaler = self.load_scaler(dname, cell_id)
        if feature_sequence is None:
            seq_arr = np.zeros((5, len(FeatureExtractor.DEFAULT_FEATURE_COLS)))
            seq_arr[:, 0] = cycle or 100
        else:
            seq_arr = np.array(feature_sequence)
            
        seq_tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0)
        
        if model_type == "ai":
            model = self.load_ai_model(dname, cell_id)
        else:
            model = self.load_pinn_model(dname, cell_id)
            
        with torch.no_grad():
            pred_scaled = model(seq_tensor).item()
        pred_cap = scaler.inverse_transform_target(np.array([pred_scaled]), "capacity")[0]
        return float(pred_cap)

    def predict_soh(
        self,
        cell_id: str,
        model_type: str = "pinn",
        feature_sequence: Optional[List[List[float]]] = None,
        cycle: Optional[int] = None
    ) -> float:
        cap = self.predict_capacity(cell_id, model_type, feature_sequence, cycle)
        q0 = self.get_initial_capacity(cell_id)
        return float(round(cap / q0, 4))

    def forecast(
        self,
        cell_id: str,
        model_type: str = "pinn",
        start_cycle: int = 100,
        horizon: int = 30
    ) -> Dict[str, Any]:
        dname = self.get_dataset_for_cell(cell_id)
        future_cycles = list(range(start_cycle, start_cycle + horizon))
        q0 = self.get_initial_capacity(cell_id)
        
        if model_type == "physics":
            phys = self.load_physics_model(dname, cell_id)
            caps = phys.predict(future_cycles)
        else:
            csv_path = self.data_dir / f"{dname}/{cell_id}.csv"
            df = pd.read_csv(csv_path)
            df_feat = self.extractor.extract_features(df)
            scaler = self.load_scaler(dname, cell_id)
            feature_cols = FeatureExtractor.DEFAULT_FEATURE_COLS
            seq_length = 5
            
            hist_idx = min(len(df_feat) - 1, max(seq_length, start_cycle))
            curr_seq = scaler.transform(df_feat.iloc[hist_idx - seq_length : hist_idx])[feature_cols].values
            
            model = self.load_pinn_model(dname, cell_id) if model_type == "pinn" else self.load_ai_model(dname, cell_id)
            caps = []
            for step in range(horizon):
                inp_t = torch.tensor(curr_seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_scaled = model(inp_t).item()
                pred_cap = scaler.inverse_transform_target(np.array([pred_scaled]), "capacity")[0]
                caps.append(float(pred_cap))
                
                next_feat = curr_seq[-1].copy()
                next_feat[0] += 1.0 / scaler.stds.get("cycle", 1.0)
                next_feat[5] = pred_scaled
                curr_seq = np.vstack([curr_seq[1:], next_feat])
                
        caps = np.array(caps)
        sohs = caps / q0
        plaus = self.plausibility_engine.evaluate_plausibility(caps, cell_id)
        
        return {
            "battery_id": cell_id,
            "cell_id": cell_id,
            "model_type": model_type,
            "start_cycle": start_cycle,
            "horizon": horizon,
            "cycles": future_cycles,
            "predicted_capacity": [round(float(c), 4) for c in caps],
            "predicted_soh": [round(float(s), 4) for s in sohs],
            "plausibility": plaus
        }

    def estimate_uncertainty(
        self,
        cell_id: str,
        feature_sequence: List[List[float]],
        n_samples: int = 100,
        confidence: float = 0.90
    ) -> Dict[str, Any]:
        dname = self.get_dataset_for_cell(cell_id)
        scaler = self.load_scaler(dname, cell_id)
        model = self.load_pinn_model(dname, cell_id)
        
        inp_t = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0)
        mean_s, std_s, low_s, high_s = predict_mc_dropout(model, inp_t, n_samples=n_samples, confidence_level=confidence)
        
        mean_cap = scaler.inverse_transform_target(mean_s, "capacity")[0]
        low_cap = scaler.inverse_transform_target(low_s, "capacity")[0]
        high_cap = scaler.inverse_transform_target(high_s, "capacity")[0]
        std_cap = float(std_s[0] * scaler.stds.get("capacity", 1.0))
        
        q0 = self.get_initial_capacity(cell_id)
        return {
            "battery_id": cell_id,
            "cell_id": cell_id,
            "mean_capacity": round(float(mean_cap), 4),
            "std_capacity": round(float(std_cap), 4),
            "lower_bound": round(float(low_cap), 4),
            "upper_bound": round(float(high_cap), 4),
            "confidence_level": confidence,
            "mean_soh": round(float(mean_cap / q0), 4)
        }
