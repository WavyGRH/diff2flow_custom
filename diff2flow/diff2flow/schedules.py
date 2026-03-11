"""
Noise Schedules for Diffusion Models.

Implements the variance-preserving (VP) and variance-exploding (VE) noise
schedules used in diffusion models like Stable Diffusion. Supports both
discrete integer timesteps and continuous interpolated timesteps — a key
requirement for the Diff2Flow timestep mapping.

Reference: Section 3.1, Equation 1 of the Diff2Flow paper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import numpy as np


class NoiseScheduleVP:
    """Variance-Preserving (VP) noise schedule.

    Defines alpha_t and sigma_t such that:
        x_t = alpha_t * x_0 + sigma_t * epsilon
    with the constraint: alpha_t^2 + sigma_t^2 = 1

    This is the schedule used by Stable Diffusion (DDPM linear beta schedule).

    Args:
        num_timesteps: Number of discrete timesteps T (default 1000 for SD).
        beta_start: Starting value of the linear beta schedule.
        beta_end: Ending value of the linear beta schedule.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Compute the linear beta schedule (as in DDPM / Stable Diffusion)
        # SD uses scaled linear schedule: betas = linspace(sqrt(beta_start), sqrt(beta_end))^2
        betas = torch.linspace(
            math.sqrt(beta_start), math.sqrt(beta_end), num_timesteps
        ) ** 2

        # alpha_bar_t = cumulative product of (1 - beta_i) for i in [0, t]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Store alpha_t = sqrt(alpha_bar_t) and sigma_t = sqrt(1 - alpha_bar_t)
        self._alpha_values = torch.sqrt(alphas_cumprod)  # shape: [T]
        self._sigma_values = torch.sqrt(1.0 - alphas_cumprod)  # shape: [T]

    @property
    def alpha_values(self) -> torch.Tensor:
        """Precomputed alpha_t for all discrete timesteps [0, T-1]."""
        return self._alpha_values

    @property
    def sigma_values(self) -> torch.Tensor:
        """Precomputed sigma_t for all discrete timesteps [0, T-1]."""
        return self._sigma_values

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t at (possibly continuous) timestep t.

        For integer timesteps, returns the exact value. For continuous
        timesteps, performs linear interpolation between nearest discrete
        neighbors — this is critical for the Diff2Flow mapping (Sec 3.2.1).

        Args:
            t: Timestep(s), can be continuous floats in [0, T-1].

        Returns:
            alpha_t values with same shape as t.
        """
        return self._interpolate(self._alpha_values, t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma_t at (possibly continuous) timestep t.

        Supports linear interpolation at continuous timesteps.

        Args:
            t: Timestep(s), can be continuous floats in [0, T-1].

        Returns:
            sigma_t values with same shape as t.
        """
        return self._interpolate(self._sigma_values, t)

    def _interpolate(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate schedule values at continuous timesteps.

        Given precomputed values at discrete timesteps [0, 1, ..., T-1],
        returns interpolated values at arbitrary continuous t.

        Args:
            values: Precomputed schedule values, shape [T].
            t: Continuous timestep(s).

        Returns:
            Interpolated values with same shape as t.
        """
        t = t.float()
        t_low = t.floor().long().clamp(0, len(values) - 1)
        t_high = t.ceil().long().clamp(0, len(values) - 1)
        frac = t - t.floor()

        # Handle edge case where t is exactly integer
        frac = frac.clamp(0.0, 1.0)

        val_low = values[t_low]
        val_high = values[t_high]

        return val_low + frac * (val_high - val_low)

    def verify_vp_constraint(self, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Verify that alpha_t^2 + sigma_t^2 ≈ 1 (VP constraint).

        Args:
            t: Timesteps to check. If None, checks all discrete timesteps.

        Returns:
            Maximum absolute deviation from 1.0.
        """
        if t is None:
            t = torch.arange(self.num_timesteps, dtype=torch.float32)
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        deviation = (alpha ** 2 + sigma ** 2 - 1.0).abs().max()
        return deviation


class NoiseScheduleVE:
    """Variance-Exploding (VE) noise schedule.

    Defines: x_t = x_0 + sigma_t * epsilon
    where alpha_t = 1 for all t and sigma_t increases monotonically.

    This is less commonly used in practice but included for completeness,
    as the Diff2Flow paper discusses both VP and VE schedules.

    Args:
        num_timesteps: Number of discrete timesteps T.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ):
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Geometric schedule for sigma
        sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_min), math.log(sigma_max), num_timesteps
            )
        )
        self._sigma_values = sigmas
        self._alpha_values = torch.ones(num_timesteps)

    @property
    def alpha_values(self) -> torch.Tensor:
        return self._alpha_values

    @property
    def sigma_values(self) -> torch.Tensor:
        return self._sigma_values

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Alpha is always 1 for VE schedule."""
        return torch.ones_like(t, dtype=torch.float32)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma_t at (possibly continuous) timestep t."""
        return self._interpolate(self._sigma_values, t)

    def _interpolate(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Same interpolation logic as VP schedule."""
        t = t.float()
        t_low = t.floor().long().clamp(0, len(values) - 1)
        t_high = t.ceil().long().clamp(0, len(values) - 1)
        frac = (t - t.floor()).clamp(0.0, 1.0)
        return values[t_low] + frac * (values[t_high] - values[t_low])
