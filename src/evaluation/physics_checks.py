from typing import Any, Dict, List, Tuple
import numpy as np

class PhysicalPlausibilityEngine:
    """Evaluates physical plausibility of battery capacity predictions."""
    
    def check_monotonicity(self, predictions: np.ndarray, threshold: float = 0.005) -> Tuple[int, float, List[int]]:
        """Check for non-physical capacity increases."""
        preds = np.asarray(predictions, dtype=np.float64)
        if len(preds) <= 1:
            return 0, 0.0, []
        deltas = preds[1:] - preds[:-1]
        violations = np.where(deltas > threshold)[0]
        count = int(len(violations))
        rate = float(count / (len(preds) - 1) * 100.0)
        return count, rate, violations.tolist()

    def check_negative_capacity(self, predictions: np.ndarray) -> Tuple[int, float]:
        """Check for non-physical negative or near-zero capacities."""
        preds = np.asarray(predictions, dtype=np.float64)
        negatives = np.where(preds <= 0.05)[0]
        count = int(len(negatives))
        rate = float(count / len(preds) * 100.0)
        return count, rate

    def check_capacity_jump(self, predictions: np.ndarray, max_jump: float = 0.08) -> Tuple[int, float]:
        """Check for sudden non-physical jumps between successive cycles."""
        preds = np.asarray(predictions, dtype=np.float64)
        if len(preds) <= 1:
            return 0, 0.0
        deltas = np.abs(preds[1:] - preds[:-1])
        jumps = np.where(deltas > max_jump)[0]
        count = int(len(jumps))
        rate = float(count / (len(preds) - 1) * 100.0)
        return count, rate

    def evaluate_plausibility(
        self,
        predictions: np.ndarray,
        cell_id: str = "Cell"
    ) -> Dict[str, Any]:
        """Run comprehensive plausibility checks and generate structured score."""
        mono_cnt, mono_rate, mono_idx = self.check_monotonicity(predictions)
        neg_cnt, neg_rate = self.check_negative_capacity(predictions)
        jump_cnt, jump_rate = self.check_capacity_jump(predictions)
        
        explanations = []
        if mono_cnt > 0:
            explanations.append(f"Monotonicity violation detected at {mono_cnt} cycle transitions ({mono_rate:.1f}%)")
        if neg_cnt > 0:
            explanations.append(f"Negative/Zero capacity violation at {neg_cnt} points")
        if jump_cnt > 0:
            explanations.append(f"Capacity discontinuity jump detected at {jump_cnt} cycles")
            
        if neg_cnt > 0 or jump_rate > 15.0 or mono_rate > 20.0:
            status = "FAIL"
        elif mono_cnt > 0 or jump_cnt > 0:
            status = "WARNING"
        else:
            status = "PASS"
            explanations.append("All physical degradation invariants fully satisfied.")
            
        plausibility_score = max(0.0, 100.0 - (mono_rate * 2.0 + neg_rate * 5.0 + jump_rate * 2.5))
        
        return {
            "status": status,
            "plausibility_score": round(plausibility_score, 1),
            "monotonicity_violations": mono_cnt,
            "monotonicity_rate": round(mono_rate, 2),
            "negative_capacity_violations": neg_cnt,
            "capacity_jump_violations": jump_cnt,
            "explanations": explanations
        }
