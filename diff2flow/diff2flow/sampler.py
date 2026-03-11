"""
Euler ODE Sampler for Flow Matching Inference.

Implements the forward Euler method for sampling from flow matching models:
  x_{i+1} = x_i + t_δ * v_θ(x_i, t_i)

where t_δ = 1/N and N is the number of function evaluations (NFE).

The sampler supports:
- Standard FM sampling (noise -> data direction)
- Variable NFE (2, 4, 10, 25, 50 steps)
- Diff2Flow trajectory traversal during inference
- Classifier-free guidance (CFG)

Reference: Section 3.1, Equation 6 of the Diff2Flow paper.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from tqdm import tqdm

from .timestep_mapping import TimestepMapper
from .interpolant_align import InterpolantAligner


class EulerSampler:
    """Forward Euler ODE sampler for FM inference.

    Integrates the learned ODE from noise (t=0) to data (t=1) using
    the forward Euler method with uniform step sizes.

    Args:
        num_steps: Number of Euler steps (NFE).
        mapper: Optional timestep mapper for Diff2Flow inference.
        aligner: Optional interpolant aligner for trajectory traversal.
        use_diff2flow: If True, performs trajectory traversal during
            inference (maps FM inputs to DM space before model evaluation).
    """

    def __init__(
        self,
        num_steps: int = 25,
        mapper: Optional[TimestepMapper] = None,
        aligner: Optional[InterpolantAligner] = None,
        use_diff2flow: bool = True,
    ):
        self.num_steps = num_steps
        self.mapper = mapper
        self.aligner = aligner
        self.use_diff2flow = use_diff2flow and (mapper is not None)

        # Uniform timestep schedule from 0 to 1
        self.timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
        self.dt = 1.0 / num_steps

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable,
        x_init: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Sample from the flow matching model via Euler integration.

        Integrates from t=0 (noise) to t=1 (data):
          x_{i+1} = x_i + dt * v_θ(x_i, t_i)

        Args:
            model_fn: Function that takes (x, t, encoder_hidden_states)
                and returns velocity (or DM prediction for Diff2Flow).
            x_init: Initial noise, shape [B, C, H, W].
            encoder_hidden_states: Text embeddings for conditioning.
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).
            show_progress: Whether to show a progress bar.

        Returns:
            Generated sample, shape [B, C, H, W].
        """
        device = x_init.device
        x = x_init.clone()

        timesteps = self.timesteps.to(device)
        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling", leave=False)

        for i in iterator:
            t_fm = timesteps[i]
            t_batch = t_fm.expand(x.shape[0])

            # Get velocity prediction
            velocity = self._get_velocity(
                model_fn, x, t_batch, encoder_hidden_states, guidance_scale
            )

            # Euler step
            x = x + self.dt * velocity

        return x

    def _get_velocity(
        self,
        model_fn: Callable,
        x_fm: torch.Tensor,
        t_fm: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """Get velocity prediction, with optional Diff2Flow traversal and CFG.

        If use_diff2flow is True:
        1. Map t_FM -> t_DM_bar
        2. Transform x_FM -> x_DM (via fx_inverse)
        3. Run model on (x_DM, t_DM_bar)
        4. Convert model output to FM velocity

        Args:
            model_fn: Model forward function.
            x_fm: Current FM state, shape [B, C, H, W].
            t_fm: Current FM timestep, shape [B].
            encoder_hidden_states: Text conditioning.
            guidance_scale: CFG scale.

        Returns:
            FM velocity, shape [B, C, H, W].
        """
        if self.use_diff2flow and self.mapper is not None and self.aligner is not None:
            # Map to DM space for the model
            t_dm = self.mapper.fm_to_dm_batched(t_fm)
            x_dm = self.aligner.fm_to_dm(x_fm, t_fm)

            model_input = x_dm
            model_t = t_dm
        else:
            model_input = x_fm
            model_t = t_fm

        # Classifier-free guidance
        if guidance_scale > 1.0 and encoder_hidden_states is not None:
            # Unconditional prediction
            null_states = torch.zeros_like(encoder_hidden_states)
            v_uncond = model_fn(model_input, model_t, null_states)
            v_cond = model_fn(model_input, model_t, encoder_hidden_states)
            velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            velocity = model_fn(
                model_input, model_t,
                encoder_hidden_states if encoder_hidden_states is not None else None,
            )

        return velocity

    def sample_with_trajectory(
        self,
        model_fn: Callable,
        x_init: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> list[torch.Tensor]:
        """Sample and return the full trajectory for visualization.

        Returns:
            List of tensors from x_0 (noise) to x_N (generated sample).
        """
        device = x_init.device
        x = x_init.clone()
        trajectory = [x.clone()]

        timesteps = self.timesteps.to(device)

        for i in range(self.num_steps):
            t_fm = timesteps[i]
            t_batch = t_fm.expand(x.shape[0])

            velocity = self._get_velocity(
                model_fn, x, t_batch, encoder_hidden_states, guidance_scale
            )

            x = x + self.dt * velocity
            trajectory.append(x.clone())

        return trajectory

    def compute_straightness(self, trajectory: list[torch.Tensor]) -> float:
        """Compute trajectory straightness metric.

        Straightness is measured as the ratio of the direct distance
        (start to end) to the total path length. A value of 1.0 means
        a perfectly straight trajectory.

        Args:
            trajectory: List of states along the trajectory.

        Returns:
            Straightness score in [0, 1].
        """
        if len(trajectory) < 2:
            return 1.0

        # Direct distance
        direct = (trajectory[-1] - trajectory[0]).flatten(1).norm(dim=1)

        # Path length
        path_length = torch.zeros(trajectory[0].shape[0], device=trajectory[0].device)
        for i in range(1, len(trajectory)):
            step_dist = (trajectory[i] - trajectory[i - 1]).flatten(1).norm(dim=1)
            path_length += step_dist

        straightness = (direct / path_length.clamp(min=1e-8)).mean().item()
        return straightness
