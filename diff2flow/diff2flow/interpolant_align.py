"""
Interpolant Alignment between Diffusion and Flow Matching trajectories.

Implements the spatial mapping fx that transforms between:
  - DM interpolant: x_DM = alpha_t * x_0 + sigma_t * epsilon
  - FM interpolant: x_FM = (1 - t_FM) * x_0 + t_FM * x_1

The key insight is that x_FM = x_DM / (alpha_t + sigma_t), which rescales
the diffusion trajectory to lie on the linear FM path.

Reference: Section 3.2.1, Equations 9-10, 13 of the Diff2Flow paper.
"""

from __future__ import annotations

import torch

from .schedules import NoiseScheduleVP
from .timestep_mapping import TimestepMapper


class InterpolantAligner:
    """Aligns DM and FM interpolants via the fx transformation.

    Given a noise schedule and timestep mapper, provides bidirectional
    transformation between DM and FM interpolant representations.

    Args:
        schedule: Noise schedule providing alpha_t and sigma_t.
        mapper: Timestep mapper providing t_DM <-> t_FM conversion.
    """

    def __init__(
        self,
        schedule: NoiseScheduleVP | None = None,
        mapper: TimestepMapper | None = None,
    ):
        if schedule is None:
            schedule = NoiseScheduleVP()
        if mapper is None:
            mapper = TimestepMapper(schedule)

        self.schedule = schedule
        self.mapper = mapper

    def dm_to_fm(
        self,
        x_dm: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> torch.Tensor:
        """Transform DM interpolant to FM interpolant.

        fx(x_DM, t_DM) = x_DM / (alpha_{t_DM} + sigma_{t_DM})

        This rescales the variance-preserving trajectory onto the
        linear FM trajectory (Eq. 10 of the paper).

        Args:
            x_dm: DM interpolant, shape [B, C, H, W] or any shape.
            t_dm: Diffusion timestep(s), shape [B] or broadcastable.

        Returns:
            FM interpolant x_FM, same shape as x_dm.
        """
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)
        scale = alpha + sigma

        # Reshape scale for broadcasting with spatial dims
        while scale.dim() < x_dm.dim():
            scale = scale.unsqueeze(-1)

        return x_dm / scale

    def fm_to_dm(
        self,
        x_fm: torch.Tensor,
        t_fm: torch.Tensor,
    ) -> torch.Tensor:
        """Transform FM interpolant back to DM interpolant.

        fx_inverse(x_FM, t_FM) = x_FM * (alpha_{t_DM} + sigma_{t_DM})

        where t_DM = ft_inverse(t_FM).

        Args:
            x_fm: FM interpolant, shape [B, C, H, W] or any shape.
            t_fm: Flow matching timestep(s), shape [B] or broadcastable.

        Returns:
            DM interpolant x_DM, same shape as x_fm.
        """
        # First map t_FM -> t_DM
        t_dm = self.mapper.fm_to_dm_batched(t_fm)

        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)
        scale = alpha + sigma

        while scale.dim() < x_fm.dim():
            scale = scale.unsqueeze(-1)

        return x_fm * scale

    @staticmethod
    def construct_dm_interpolant(
        x_0: torch.Tensor,
        epsilon: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the DM interpolant from clean data and noise.

        x_DM = alpha_t * x_0 + sigma_t * epsilon

        Args:
            x_0: Clean data samples, shape [B, C, H, W].
            epsilon: Gaussian noise, shape [B, C, H, W].
            alpha_t: Schedule alpha values, shape [B] (will be broadcast).
            sigma_t: Schedule sigma values, shape [B] (will be broadcast).

        Returns:
            DM interpolant x_DM, shape [B, C, H, W].
        """
        # Reshape for broadcasting
        while alpha_t.dim() < x_0.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        while sigma_t.dim() < x_0.dim():
            sigma_t = sigma_t.unsqueeze(-1)

        return alpha_t * x_0 + sigma_t * epsilon

    @staticmethod
    def construct_fm_interpolant(
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t_fm: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the FM linear interpolant.

        x_FM = (1 - t_FM) * x_0 + t_FM * x_1

        where x_0 is noise and x_1 is data (FM convention).

        Args:
            x_0: Noise samples (FM x_0), shape [B, C, H, W].
            x_1: Clean data (FM x_1), shape [B, C, H, W].
            t_fm: FM timesteps, shape [B] (will be broadcast).

        Returns:
            FM interpolant x_FM, shape [B, C, H, W].
        """
        while t_fm.dim() < x_0.dim():
            t_fm = t_fm.unsqueeze(-1)

        return (1.0 - t_fm) * x_0 + t_fm * x_1

    def verify_alignment(
        self,
        x_0: torch.Tensor,
        epsilon: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> dict:
        """Verify that DM -> FM mapping produces valid FM interpolant.

        Constructs both interpolants independently and checks they match
        after transformation.

        Returns:
            Dictionary with alignment error metrics.
        """
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)
        t_fm = self.mapper.dm_to_fm(t_dm)

        # Construct DM interpolant
        x_dm = self.construct_dm_interpolant(x_0, epsilon, alpha, sigma)

        # Transform to FM space
        x_fm_from_dm = self.dm_to_fm(x_dm, t_dm)

        # Construct FM interpolant directly
        # FM convention: x_0 = noise (epsilon), x_1 = data (x_0 in DM notation)
        x_fm_direct = self.construct_fm_interpolant(epsilon, x_0, t_fm)

        error = (x_fm_from_dm - x_fm_direct).abs()

        return {
            "max_error": error.max().item(),
            "mean_error": error.mean().item(),
            "t_dm": t_dm,
            "t_fm": t_fm,
        }
