import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

class ExperimentTracker:
    """Logs experiment metadata, metrics history, and persists training logs."""
    
    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics}
        self.history.append(row)
        
    def save(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if metadata:
            self.metadata.update(metadata)
            
        # Save training_log.csv
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_dir / "training_log.csv", index=False)
        
        # Save metrics.json
        full_data = {
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "final_metrics": self.history[-1] if self.history else {}
        }
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2)
