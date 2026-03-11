"""
Training Loop for Diff2Flow.

Supports three training objectives:
  1. Diffusion: Standard v-prediction / epsilon-prediction loss
  2. FM (naive): Direct flow matching velocity loss (baseline)
  3. Diff2Flow: FM loss with trajectory alignment (proposed method)

Features: mixed precision, gradient accumulation, WandB logging,
checkpointing, and configurable via YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .converter import Diff2FlowConverter
from .schedules import NoiseScheduleVP

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration.

    Attributes:
        objective: Training objective ('diffusion', 'fm', 'diff2flow').
        learning_rate: Initial learning rate.
        num_iterations: Total training iterations.
        batch_size: Training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        use_fp16: Enable mixed precision training.
        save_every: Checkpoint save interval (iterations).
        log_every: Logging interval (iterations).
        output_dir: Directory for checkpoints and logs.
        max_grad_norm: Maximum gradient norm for clipping.
        lr_scheduler: Learning rate scheduler type ('constant', 'cosine', 'linear').
        warmup_steps: Number of warmup steps.
    """
    objective: str = "diff2flow"
    learning_rate: float = 1e-5
    num_iterations: int = 20000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    use_fp16: bool = True
    save_every: int = 2000
    log_every: int = 100
    output_dir: str = "outputs"
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 500


class Diff2FlowTrainer:
    """Training loop for Diff2Flow finetuning.

    Handles the complete training pipeline including data loading,
    forward/backward passes, optimization, and checkpointing.

    Args:
        model: The Diff2FlowModel (with or without LoRA).
        config: Training configuration.
        converter: Diff2FlowConverter instance (for Diff2Flow objective).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        converter: Optional[Diff2FlowConverter] = None,
    ):
        self.model = model
        self.config = config

        # Initialize converter based on objective
        if config.objective == "diff2flow" and converter is None:
            self.converter = Diff2FlowConverter(parameterization="v")
        else:
            self.converter = converter

        self.schedule = NoiseScheduleVP()
        self.device = next(model.parameters()).device if list(model.parameters()) else "cpu"

        # Setup will be done in train()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def setup_training(self):
        """Initialize optimizer, scheduler, and GradScaler."""
        # Optimizer: only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_iterations,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.config.num_iterations,
            )
        else:
            self.scheduler = None

        # Mixed precision
        if self.config.use_fp16 and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def train(self, dataloader: DataLoader):
        """Run the training loop.

        Args:
            dataloader: DataLoader yielding training batches.
                Each batch should be a dict with at least:
                - 'latent': Latent code, shape [B, C, H, W]
                - 'encoder_hidden_states': Text condition, shape [B, 77, dim]
                  (optional, defaults to null embedding)
                - 'context': Additional context for concatenation (optional)
        """
        self.setup_training()
        self.model.train()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        data_iter = iter(dataloader)
        pbar = tqdm(range(1, self.config.num_iterations + 1), desc="Training")
        running_loss = 0.0

        for step in pbar:
            # Get batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            loss = self._training_step(batch)
            running_loss += loss

            # Optimizer step (with gradient accumulation)
            if step % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            # Logging
            if step % self.config.log_every == 0:
                avg_loss = running_loss / self.config.log_every
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                running_loss = 0.0

            # Checkpointing
            if step % self.config.save_every == 0:
                self._save_checkpoint(step, output_dir)

        # Final save
        self._save_checkpoint(self.config.num_iterations, output_dir)
        logger.info("Training complete.")

    def _training_step(self, batch: dict) -> float:
        """Single training step.

        Implements the forward pass for the selected objective
        (diffusion, FM, or Diff2Flow).

        Args:
            batch: Training batch dictionary.

        Returns:
            Loss value as a float.
        """
        device = self.device

        # Extract data from batch
        x_data = batch["latent"].to(device)
        encoder_hidden_states = batch.get("encoder_hidden_states")
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device)

        context = batch.get("context")
        if context is not None:
            context = context.to(device)

        B = x_data.shape[0]

        # Sample noise
        epsilon = torch.randn_like(x_data)

        # Sample timestep
        t_fm = torch.rand(B, device=device)

        # Forward pass based on objective
        use_amp = self.scaler is not None
        with torch.amp.autocast("cuda", enabled=use_amp):
            if self.config.objective == "diff2flow":
                loss = self._diff2flow_step(
                    x_data, epsilon, t_fm, encoder_hidden_states, context
                )
            elif self.config.objective == "fm":
                loss = self._fm_step(
                    x_data, epsilon, t_fm, encoder_hidden_states, context
                )
            elif self.config.objective == "diffusion":
                loss = self._diffusion_step(
                    x_data, epsilon, t_fm, encoder_hidden_states, context
                )
            else:
                raise ValueError(f"Unknown objective: {self.config.objective}")

        # Backward pass
        scaled_loss = loss / self.config.gradient_accumulation_steps
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
            if hasattr(self, "_current_step") and self._current_step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
        else:
            scaled_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )

        return loss.item()

    def _diff2flow_step(
        self, x_data, epsilon, t_fm, encoder_hidden_states, context
    ) -> torch.Tensor:
        """Diff2Flow training step (proposed method).

        1. Prepare FM training sample (with trajectory alignment)
        2. Map to DM space for model input
        3. Run model in its native parameterization
        4. Derive FM velocity from model output
        5. Compute loss against target velocity
        """
        sample = self.converter.prepare_training_sample(x_data, epsilon, t_fm)

        # Run model on DM-space inputs
        model_output = self.model(
            sample.x_dm,
            sample.t_dm_bar,
            encoder_hidden_states=encoder_hidden_states,
            context=context,
        )

        # Compute Diff2Flow loss
        loss = self.converter.compute_loss(model_output, sample)
        return loss

    def _fm_step(
        self, x_data, epsilon, t_fm, encoder_hidden_states, context
    ) -> torch.Tensor:
        """Naive FM training step (baseline).

        Directly applies FM loss without trajectory alignment.
        The paper shows this performs poorly, especially with LoRA.
        """
        # Construct FM interpolant
        t_fm_expanded = t_fm
        while t_fm_expanded.dim() < x_data.dim():
            t_fm_expanded = t_fm_expanded.unsqueeze(-1)

        x_fm = (1.0 - t_fm_expanded) * epsilon + t_fm_expanded * x_data

        # Model predicts velocity directly
        model_output = self.model(
            x_fm, t_fm,
            encoder_hidden_states=encoder_hidden_states,
            context=context,
        )

        # Target velocity
        v_target = x_data - epsilon
        loss = torch.mean((model_output - v_target) ** 2)
        return loss

    def _diffusion_step(
        self, x_data, epsilon, t_fm, encoder_hidden_states, context
    ) -> torch.Tensor:
        """Standard diffusion training step (baseline).

        Uses the original DM v-prediction loss.
        """
        # Map FM timestep to DM timestep
        t_dm_int = (t_fm * (self.schedule.num_timesteps - 1)).long()
        alpha = self.schedule.alpha(t_dm_int.float())
        sigma = self.schedule.sigma(t_dm_int.float())

        # Construct DM interpolant
        while alpha.dim() < x_data.dim():
            alpha = alpha.unsqueeze(-1)
        while sigma.dim() < x_data.dim():
            sigma = sigma.unsqueeze(-1)

        x_dm = alpha * x_data + sigma * epsilon

        # Model predicts v
        model_output = self.model(
            x_dm, t_dm_int.float(),
            encoder_hidden_states=encoder_hidden_states,
            context=context,
        )

        # v-prediction target
        v_target = alpha * epsilon - sigma * x_data
        loss = torch.mean((model_output - v_target) ** 2)
        return loss

    def _save_checkpoint(self, step: int, output_dir: Path):
        """Save a training checkpoint."""
        checkpoint_path = output_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state = {
            "step": step,
            "model_state_dict": {
                k: v for k, v in self.model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }

        torch.save(state, checkpoint_path / "training_state.pt")
        logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")
