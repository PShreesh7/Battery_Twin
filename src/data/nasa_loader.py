from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("nasa_loader")

class NASADataLoader:
    """Loader for NASA PCoE Battery Dataset."""
    
    def __init__(self, raw_dir: Union[str, Path] = "data/raw/nasa"):
        self.raw_dir = Path(raw_dir)
        
    def load_cell(self, cell_id: str) -> pd.DataFrame:
        """Load cycle-level dataset for a specific NASA cell."""
        csv_path = self.raw_dir / f"{cell_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"NASA cell dataset not found at {csv_path}. Run scripts/download_data.py first.")
        df = pd.read_csv(csv_path)
        df["dataset"] = "nasa_pcoe"
        return df

    def load_all(self, cell_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Load multiple cells and concatenate into a unified dataframe."""
        if cell_ids is None:
            cell_ids = ["B0005", "B0006", "B0007", "B0018"]
        dfs = [self.load_cell(cid) for cid in cell_ids]
        return pd.concat(dfs, ignore_index=True)
