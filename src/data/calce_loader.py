from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("calce_loader")

class CALCEDataLoader:
    """Loader for CALCE CS2 Battery Dataset."""
    
    def __init__(self, raw_dir: Union[str, Path] = "data/raw/calce"):
        self.raw_dir = Path(raw_dir)
        
    def load_cell(self, cell_id: str) -> pd.DataFrame:
        """Load cycle-level dataset for a specific CALCE cell."""
        csv_path = self.raw_dir / f"{cell_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CALCE cell dataset not found at {csv_path}. Run scripts/download_data.py first.")
        df = pd.read_csv(csv_path)
        df["dataset"] = "calce_cs2"
        return df

    def load_all(self, cell_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Load multiple CALCE cells into a unified dataframe."""
        if cell_ids is None:
            cell_ids = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
        dfs = [self.load_cell(cid) for cid in cell_ids]
        return pd.concat(dfs, ignore_index=True)
