"""
Unit tests for timestep mapping.

Verifies:
- Boundary conditions: ft(0) = 1.0, ft(T) ≈ 0.0
- Round-trip invertibility: ft_inverse(ft(t)) ≈ t
- Monotonic behavior (decreasing)
- Consistency between scalar and batch operations
"""

import pytest
import torch

from diff2flow.schedules import NoiseScheduleVP
from diff2flow.timestep_mapping import TimestepMapper


class TestTimestepMapper:
    """Tests for the ft / ft_inverse timestep mapping."""

    def setup_method(self):
        self.schedule = NoiseScheduleVP(num_timesteps=1000)
        self.mapper = TimestepMapper(self.schedule)

    def test_boundary_t0_maps_to_1(self):
        """DM t=0 (data) should map to FM t=1 (data)."""
        t_dm = torch.tensor([0.0])
        t_fm = self.mapper.dm_to_fm(t_dm).item()

        # At t=0: alpha≈1, sigma≈0, so ft = alpha/(alpha+sigma) ≈ 1
        # Note: SD's scaled linear schedule gives alpha(0) ≈ 0.9992, sigma(0) ≈ 0.029
        assert t_fm > 0.95, f"ft(0) should be ~1.0, got {t_fm}"

    def test_boundary_tmax_maps_near_0(self):
        """DM t=T-1 (noise) should map to FM t≈0 (noise)."""
        t_dm = torch.tensor([999.0])
        t_fm = self.mapper.dm_to_fm(t_dm).item()

        # At t=999: alpha≈0, sigma≈1, so ft ≈ 0
        assert t_fm < 0.1, f"ft(999) should be ~0, got {t_fm}"

    def test_monotonically_decreasing(self):
        """ft(t) should be monotonically decreasing (higher DM t → lower FM t)."""
        t_dm = torch.arange(1000, dtype=torch.float32)
        t_fm = self.mapper.dm_to_fm(t_dm)

        diffs = t_fm[1:] - t_fm[:-1]
        assert (diffs <= 0).all(), "ft must be monotonically decreasing"

    def test_round_trip_discrete(self):
        """ft_inverse(ft(t)) ≈ t for discrete timesteps."""
        test_timesteps = torch.tensor([0.0, 100.0, 250.0, 500.0, 750.0, 999.0])

        for t_dm in test_timesteps:
            t_dm_tensor = t_dm.unsqueeze(0)
            t_fm = self.mapper.dm_to_fm(t_dm_tensor)
            t_dm_recovered = self.mapper.fm_to_dm(t_fm).item()

            assert abs(t_dm_recovered - t_dm.item()) < 2.0, (
                f"Round-trip failed: {t_dm.item()} -> {t_fm.item()} -> {t_dm_recovered}"
            )

    def test_round_trip_batched(self):
        """Batched inverse should match scalar inverse."""
        test_timesteps = torch.tensor([0.0, 100.0, 500.0, 999.0])
        t_fm = self.mapper.dm_to_fm(test_timesteps)

        t_dm_scalar = self.mapper.fm_to_dm(t_fm)
        t_dm_batched = self.mapper.fm_to_dm_batched(t_fm)

        assert torch.allclose(t_dm_scalar, t_dm_batched, atol=1.0), (
            f"Scalar and batched inverse differ:\n"
            f"  Scalar:  {t_dm_scalar}\n"
            f"  Batched: {t_dm_batched}"
        )

    def test_fm_range(self):
        """All FM timesteps should be in [0, 1]."""
        t_dm = torch.arange(1000, dtype=torch.float32)
        t_fm = self.mapper.dm_to_fm(t_dm)

        assert t_fm.min() >= 0.0, f"FM timesteps below 0: min = {t_fm.min()}"
        assert t_fm.max() <= 1.0, f"FM timesteps above 1: max = {t_fm.max()}"

    def test_inverse_boundary_values(self):
        """Inverse mapping at boundaries should return boundary DM values."""
        # FM t=1 (data) should map to DM t≈0 (data)
        t_fm_1 = torch.tensor([0.999])
        t_dm = self.mapper.fm_to_dm(t_fm_1).item()
        assert t_dm < 5.0, f"ft_inverse(~1) should be ~0, got {t_dm}"

        # FM t=0 (noise) should map to DM t≈999 (noise)
        t_fm_0 = torch.tensor([0.001])
        t_dm = self.mapper.fm_to_dm(t_fm_0).item()
        assert t_dm > 990.0, f"ft_inverse(~0) should be ~999, got {t_dm}"

    def test_precomputed_table_shape(self):
        """Precomputed table should have correct shape."""
        assert self.mapper.t_fm_table.shape == (1000,), (
            f"Expected table shape (1000,), got {self.mapper.t_fm_table.shape}"
        )

    def test_continuous_timestep_intermediate(self):
        """Continuous DM timesteps (e.g., 500.5) should produce reasonable FM values."""
        t_dm_int = torch.tensor([500.0])
        t_dm_half = torch.tensor([500.5])
        t_dm_next = torch.tensor([501.0])

        fm_int = self.mapper.dm_to_fm(t_dm_int).item()
        fm_half = self.mapper.dm_to_fm(t_dm_half).item()
        fm_next = self.mapper.dm_to_fm(t_dm_next).item()

        assert fm_next <= fm_half <= fm_int, (
            f"Continuous interpolation failed: ft(500)={fm_int}, "
            f"ft(500.5)={fm_half}, ft(501)={fm_next}"
        )
