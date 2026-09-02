import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.early_stopping import EarlyStopping
from src.training.experiment_tracker import ExperimentTracker
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("trainer")

class BatteryModelTrainer:
    """Unified Training Pipeline for AI-Only and PINN Models."""
    
    def __init__(
        self,
        model: nn.Module,
        experiment_id: str,
        output_dir: str,
        is_pinn: bool = False,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        epochs: int = 150,
        patience: int = 25,
        device: str = "cpu",
        seed: int = 42
    ):
        set_seed(seed)
        self.model = model
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_pinn = is_pinn
        self.epochs = epochs
        self.device = torch.device(device if (torch.cuda.is_available() and device != "cpu") else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.early_stopping = EarlyStopping(patience=patience)
        self.tracker = ExperimentTracker(experiment_id, str(self.output_dir))
        self.mse_criterion = nn.MSELoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_data_loss, total_phys_loss = 0.0, 0.0, 0.0
        n_batches = 0
        
        for X_batch, y_batch, cycles_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            cycles_batch = cycles_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            
            if self.is_pinn and hasattr(self.model, "compute_loss"):
                loss, l_data, l_phys, _ = self.model.compute_loss(y_pred, y_batch, cycles_batch)
            else:
                loss = self.mse_criterion(y_pred, y_batch)
                l_data = loss
                l_phys = torch.tensor(0.0)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += l_data.item()
            total_phys_loss += l_phys.item()
            n_batches += 1
            
        return {
            "train_loss": total_loss / max(1, n_batches),
            "train_data_loss": total_data_loss / max(1, n_batches),
            "train_phys_loss": total_phys_loss / max(1, n_batches)
        }
        
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_data_loss, total_phys_loss = 0.0, 0.0, 0.0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch, cycles_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                cycles_batch = cycles_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                
                if self.is_pinn and hasattr(self.model, "compute_loss"):
                    loss, l_data, l_phys, _ = self.model.compute_loss(y_pred, y_batch, cycles_batch)
                else:
                    loss = self.mse_criterion(y_pred, y_batch)
                    l_data = loss
                    l_phys = torch.tensor(0.0)
                    
                total_loss += loss.item()
                total_data_loss += l_data.item()
                total_phys_loss += l_phys.item()
                n_batches += 1
                
        return {
            "val_loss": total_loss / max(1, n_batches),
            "val_data_loss": total_data_loss / max(1, n_batches),
            "val_phys_loss": total_phys_loss / max(1, n_batches)
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        logger.info(f"Starting training for {self.experiment_id} on {self.device} (is_pinn={self.is_pinn})")
        best_val_loss = float("inf")
        
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(val_metrics["val_loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            combined_metrics = {
                **train_metrics,
                **val_metrics,
                "lr": current_lr
            }
            self.tracker.log_epoch(epoch, combined_metrics)
            
            # Checkpoint best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
                
            # Save last model
            torch.save(self.model.state_dict(), self.output_dir / "last_model.pt")
            
            if epoch % 25 == 0 or epoch == self.epochs:
                logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_metrics['train_loss']:.5f}, Val Loss: {val_metrics['val_loss']:.5f}")
                
            if self.early_stopping(val_metrics["val_loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch}!")
                break
                
        # Load best model weights
        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt", weights_only=True))
        self.tracker.save({"best_val_loss": best_val_loss})
        logger.info(f"Training completed. Best Val Loss: {best_val_loss:.5f}")
        return {"best_val_loss": best_val_loss}
