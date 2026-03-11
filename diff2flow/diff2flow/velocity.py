"""
Velocity Derivation from Diffusion Model Predictions.

Converts a pre-trained diffusion model's output (v-prediction or
epsilon-prediction) into the flow matching velocity field. This is
the core "objective change" that makes Diff2Flow work.

Key equations (for v-prediction):
  x̂_0 = alpha_t * x_DM - sigma_t * v_pred    (estimated clean image)
  x̂_T = sigma_t * x_DM + alpha_t * v_pred    (estimated noise)
  v_FM = x̂_0 - x̂_T                           (FM velocity = data - noise)

Reference: Section 3.2.2, Equations 15-17 of the Diff2Flow paper.
"""

from __future__ import annotations

from enum import Enum

import torch

from .schedules import NoiseScheduleVP


class Parameterization(Enum):
    """Diffusion model parameterization type."""
    V_PREDICTION = "v"
    EPSILON = "epsilon"
    X0 = "x0"


class VelocityDeriver:
    """Derives FM velocity from diffusion model predictions.

    Supports multiple parameterization types (v-prediction, epsilon,
    direct x0 prediction) and converts them to FM velocity fields.

    Args:
        schedule: Noise schedule for alpha_t / sigma_t lookups.
        parameterization: Which quantity the DM model predicts.
    """

    def __init__(
        self,
        schedule: NoiseScheduleVP | None = None,
        parameterization: Parameterization = Parameterization.V_PREDICTION,
    ):
        if schedule is None:
            schedule = NoiseScheduleVP()
        self.schedule = schedule
        self.parameterization = parameterization

    def derive_velocity(
        self,
        model_output: torch.Tensor,
        x_dm: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> torch.Tensor:
        """Derive FM velocity from diffusion model output.

        Dispatches to the appropriate derivation method based on the
        model's parameterization type.

        Args:
            model_output: Raw output from the diffusion model.
                - v-prediction: v_θ(x_DM, t_DM)
                - epsilon: ε_θ(x_DM, t_DM)
                - x0: x0_θ(x_DM, t_DM)
            x_dm: Current DM interpolant, shape [B, C, H, W].
            t_dm: DM timestep(s), shape [B].

        Returns:
            FM velocity v_FM, shape [B, C, H, W].
        """
        if self.parameterization == Parameterization.V_PREDICTION:
            return self._velocity_from_v_pred(model_output, x_dm, t_dm)
        elif self.parameterization == Parameterization.EPSILON:
            return self._velocity_from_eps_pred(model_output, x_dm, t_dm)
        elif self.parameterization == Parameterization.X0:
            return self._velocity_from_x0_pred(model_output, x_dm, t_dm)
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

    def _velocity_from_v_pred(
        self,
        v_pred: torch.Tensor,
        x_dm: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> torch.Tensor:
        """Derive FM velocity from v-prediction.

        v-prediction is defined as: v = alpha_t * epsilon - sigma_t * x_0

        From this, we can recover:
          x̂_0 = alpha_t * x_DM - sigma_t * v_pred     (Eq. 15)
          x̂_T = sigma_t * x_DM + alpha_t * v_pred     (Eq. 16)
          v_FM = x̂_0 - x̂_T                           (Eq. 17)

        Note: In FM convention, velocity = x_1 - x_0 = data - noise.

        Args:
            v_pred: Model's v-prediction, shape [B, C, H, W].
            x_dm: DM interpolant, shape [B, C, H, W].
            t_dm: DM timestep(s), shape [B].

        Returns:
            FM velocity, shape [B, C, H, W].
        """
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)

        # Reshape for broadcasting
        while alpha.dim() < x_dm.dim():
            alpha = alpha.unsqueeze(-1)
        while sigma.dim() < x_dm.dim():
            sigma = sigma.unsqueeze(-1)

        # Eq. 15: Estimate clean data
        x0_hat = alpha * x_dm - sigma * v_pred

        # Eq. 16: Estimate noise
        xT_hat = sigma * x_dm + alpha * v_pred

        # Eq. 17: FM velocity = data - noise (x_1 - x_0 in FM convention)
        v_fm = x0_hat - xT_hat

        return v_fm

    def _velocity_from_eps_pred(
        self,
        eps_pred: torch.Tensor,
        x_dm: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> torch.Tensor:
        """Derive FM velocity from epsilon-prediction.

        From x_DM = alpha_t * x_0 + sigma_t * epsilon:
          x̂_0 = (x_DM - sigma_t * eps_pred) / alpha_t
          x̂_T = eps_pred  (the noise itself)
          v_FM = x̂_0 - x̂_T

        Args:
            eps_pred: Model's noise prediction, shape [B, C, H, W].
            x_dm: DM interpolant, shape [B, C, H, W].
            t_dm: DM timestep(s), shape [B].

        Returns:
            FM velocity, shape [B, C, H, W].
        """
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)

        while alpha.dim() < x_dm.dim():
            alpha = alpha.unsqueeze(-1)
        while sigma.dim() < x_dm.dim():
            sigma = sigma.unsqueeze(-1)

        # Estimate clean data from epsilon prediction
        x0_hat = (x_dm - sigma * eps_pred) / alpha.clamp(min=1e-8)

        # Noise estimate is just the predicted epsilon
        xT_hat = eps_pred

        # FM velocity
        v_fm = x0_hat - xT_hat

        return v_fm

    def _velocity_from_x0_pred(
        self,
        x0_pred: torch.Tensor,
        x_dm: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> torch.Tensor:
        """Derive FM velocity from direct x0-prediction.

        From x_DM = alpha_t * x_0 + sigma_t * epsilon:
          x̂_0 = x0_pred
          x̂_T = (x_DM - alpha_t * x0_pred) / sigma_t
          v_FM = x̂_0 - x̂_T

        Args:
            x0_pred: Model's clean image prediction, shape [B, C, H, W].
            x_dm: DM interpolant, shape [B, C, H, W].
            t_dm: DM timestep(s), shape [B].

        Returns:
            FM velocity, shape [B, C, H, W].
        """
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)

        while alpha.dim() < x_dm.dim():
            alpha = alpha.unsqueeze(-1)
        while sigma.dim() < x_dm.dim():
            sigma = sigma.unsqueeze(-1)

        x0_hat = x0_pred
        xT_hat = (x_dm - alpha * x0_pred) / sigma.clamp(min=1e-8)

        v_fm = x0_hat - xT_hat

        return v_fm

    def compute_fm_loss(
        self,
        predicted_velocity: torch.Tensor,
        x_data: torch.Tensor,
        x_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the standard Flow Matching loss.

        L_FM = ||v_θ(x_t, t) - (x_1 - x_0)||²

        where the target velocity is simply data minus noise.

        Args:
            predicted_velocity: Model's velocity prediction, shape [B, C, H, W].
            x_data: Clean data samples (x_1 in FM), shape [B, C, H, W].
            x_noise: Noise samples (x_0 in FM), shape [B, C, H, W].

        Returns:
            Scalar loss value.
        """
        target_velocity = x_data - x_noise
        loss = torch.mean((predicted_velocity - target_velocity) ** 2)
        return loss
