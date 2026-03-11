"""
Timestep Mapping between Diffusion and Flow Matching.

Implements the bidirectional mapping ft that converts between:
  - Diffusion discrete timesteps: t_DM ∈ Z ∩ [0, T]  (t=0 is data, t=T is noise)
  - Flow matching continuous timesteps: t_FM ∈ [0, 1]  (t=0 is noise, t=1 is data)

Key equations from the paper:
  ft(t_DM) = alpha_{t_DM} / (alpha_{t_DM} + sigma_{t_DM})     [Eq. 11]
  ft_inverse uses piecewise linear interpolation                 [Eq. 12]

Reference: Section 3.2.1 of the Diff2Flow paper.
"""

from __future__ import annotations

from typing import Optional

import torch

from .schedules import NoiseScheduleVP


class TimestepMapper:
    """Bidirectional mapping between DM and FM timesteps.

    Precomputes the full mapping table for all discrete DM timesteps and
    provides efficient forward/inverse lookups with support for continuous
    timesteps via piecewise linear interpolation.

    Args:
        schedule: Noise schedule providing alpha_t and sigma_t.
            Defaults to the VP schedule used by Stable Diffusion.
    """

    def __init__(self, schedule: Optional[NoiseScheduleVP] = None):
        if schedule is None:
            schedule = NoiseScheduleVP()
        self.schedule = schedule
        self.num_timesteps = schedule.num_timesteps

        # Precompute the mapping table: t_DM -> t_FM for all discrete steps
        # DM convention: t=0 is data, t=T-1 is noise
        # FM convention: t=0 is noise, t=1 is data
        # So mapping should satisfy: ft(0) -> 1.0 (data), ft(T-1) -> ~0.0 (noise)
        t_dm_discrete = torch.arange(self.num_timesteps, dtype=torch.float32)
        alpha = schedule.alpha(t_dm_discrete)
        sigma = schedule.sigma(t_dm_discrete)

        # Eq. 11: t_FM = alpha_t / (alpha_t + sigma_t)
        # Note the reversal: DM t=0 (data, high alpha) maps to FM t=1 (data)
        self._t_fm_table = alpha / (alpha + sigma)  # shape: [T]

        # Store for inverse lookup
        # The table is monotonically decreasing (high t_FM at low t_DM)
        self._t_dm_table = t_dm_discrete

    @property
    def t_fm_table(self) -> torch.Tensor:
        """Precomputed t_FM values for each discrete t_DM. Shape: [T]."""
        return self._t_fm_table

    def dm_to_fm(self, t_dm: torch.Tensor) -> torch.Tensor:
        """Map diffusion timesteps to flow matching timesteps.

        ft(t_DM) = alpha_{t_DM} / (alpha_{t_DM} + sigma_{t_DM})

        Supports continuous t_DM via linear interpolation of the schedule
        values (as described in Section 3.2.1 of the paper).

        Args:
            t_dm: Diffusion timestep(s), shape [...], values in [0, T-1].

        Returns:
            Flow matching timestep(s), shape [...], values in [0, 1].
        """
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)
        return alpha / (alpha + sigma)

    def fm_to_dm(self, t_fm: torch.Tensor) -> torch.Tensor:
        """Map flow matching timesteps to diffusion timesteps (inverse).

        Uses piecewise linear interpolation between discrete neighbors
        as described in Eq. 12 of the paper.

        Args:
            t_fm: Flow matching timestep(s), shape [...], values in [0, 1].

        Returns:
            Continuous diffusion timestep(s), shape [...], values in [0, T-1].
        """
        t_fm = t_fm.float()
        original_shape = t_fm.shape
        t_fm_flat = t_fm.reshape(-1)

        # The mapping table is monotonically DECREASING:
        # t_fm_table[0] ≈ 1.0 (data end), t_fm_table[T-1] ≈ 0.0 (noise end)
        # We need to find the two nearest discrete neighbors for each query

        result = torch.zeros_like(t_fm_flat)

        for i, t in enumerate(t_fm_flat):
            result[i] = self._inverse_single(t.item())

        return result.reshape(original_shape)

    def _inverse_single(self, t_fm: float) -> float:
        """Inverse mapping for a single t_FM value.

        Finds the two discrete t_DM neighbors whose f_t values bracket
        the given t_FM, then linearly interpolates.
        """
        table = self._t_fm_table

        # Handle boundary cases
        if t_fm >= table[0].item():
            return 0.0
        if t_fm <= table[-1].item():
            return float(self.num_timesteps - 1)

        # Find bracketing indices in the monotonically decreasing table
        # table[idx1] >= t_fm >= table[idx2] where idx1 < idx2
        # Use searchsorted on the reversed table (since it needs ascending)
        table_rev = table.flip(0)
        idx_rev = torch.searchsorted(table_rev, torch.tensor(t_fm))
        idx_rev = idx_rev.clamp(1, len(table) - 1)

        # Convert back to original indexing
        idx2 = self.num_timesteps - idx_rev.item()
        idx1 = idx2 - 1 if idx2 > 0 else 0

        # Clamp indices
        idx1 = max(0, min(idx1, self.num_timesteps - 1))
        idx2 = max(0, min(idx2, self.num_timesteps - 1))

        if idx1 == idx2:
            return float(idx1)

        # Linear interpolation: Eq. 12
        t_fm1 = table[idx1].item()
        t_fm2 = table[idx2].item()

        if abs(t_fm1 - t_fm2) < 1e-10:
            return float(idx1)

        # Interpolation fraction
        frac = (t_fm - t_fm1) / (t_fm2 - t_fm1)
        t_dm_bar = idx1 + frac * (idx2 - idx1)

        return t_dm_bar

    def fm_to_dm_batched(self, t_fm: torch.Tensor) -> torch.Tensor:
        """Vectorized inverse mapping (faster for large batches).

        Uses a fully vectorized approach with torch.searchsorted.

        Args:
            t_fm: Flow matching timestep(s), shape [...].

        Returns:
            Continuous diffusion timestep(s), shape [...].
        """
        original_shape = t_fm.shape
        t_fm_flat = t_fm.reshape(-1).float()

        # Reverse table for ascending order (needed by searchsorted)
        table_rev = self._t_fm_table.flip(0)

        # Find insertion points in ascending table
        idx_rev = torch.searchsorted(table_rev, t_fm_flat)
        idx_rev = idx_rev.clamp(1, len(table_rev) - 1)

        # Convert to original descending indices
        idx2 = self.num_timesteps - idx_rev
        idx1 = idx2 - 1

        # Clamp
        idx1 = idx1.clamp(0, self.num_timesteps - 1)
        idx2 = idx2.clamp(0, self.num_timesteps - 1)

        # Get bracketing t_FM values
        t_fm1 = self._t_fm_table[idx1]
        t_fm2 = self._t_fm_table[idx2]

        # Linear interpolation
        denom = t_fm2 - t_fm1
        # Avoid division by zero
        denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
        frac = (t_fm_flat - t_fm1) / denom

        # Clamp frac to [0, 1]
        frac = frac.clamp(0.0, 1.0)

        t_dm_bar = idx1.float() + frac * (idx2 - idx1).float()

        # Handle boundary clamps
        t_dm_bar = t_dm_bar.clamp(0.0, float(self.num_timesteps - 1))

        return t_dm_bar.reshape(original_shape)
