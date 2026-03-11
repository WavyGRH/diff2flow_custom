"""
Unit tests for interpolant alignment.

Verifies:
- DM → FM transformation produces valid FM interpolant
- FM → DM round-trip preserves the original
- Boundary conditions at t=0 (noise) and t=1 (data)
- Shape consistency across transformations
"""

import pytest
import torch

from diff2flow.schedules import NoiseScheduleVP
from diff2flow.timestep_mapping import TimestepMapper
from diff2flow.interpolant_align import InterpolantAligner


class TestInterpolantAligner:
    """Tests for the fx / fx_inverse interpolant alignment."""

    def setup_method(self):
        self.schedule = NoiseScheduleVP(num_timesteps=1000)
        self.mapper = TimestepMapper(self.schedule)
        self.aligner = InterpolantAligner(self.schedule, self.mapper)

        # Create test data
        torch.manual_seed(42)
        self.x_0 = torch.randn(2, 4, 8, 8)  # Clean data (DM x_0 = FM x_1)
        self.epsilon = torch.randn(2, 4, 8, 8)  # Noise (DM x_T = FM x_0)

    def test_dm_to_fm_shape(self):
        """Output shape should match input shape."""
        t_dm = torch.tensor([100.0, 500.0])
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)
        x_dm = self.aligner.construct_dm_interpolant(self.x_0, self.epsilon, alpha, sigma)

        x_fm = self.aligner.dm_to_fm(x_dm, t_dm)
        assert x_fm.shape == x_dm.shape, f"Shape mismatch: {x_fm.shape} != {x_dm.shape}"

    def test_alignment_at_data_end(self):
        """At t_DM=0 (data end): transformed x_FM should ≈ clean data."""
        t_dm = torch.tensor([0.0, 0.0])
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)
        x_dm = self.aligner.construct_dm_interpolant(self.x_0, self.epsilon, alpha, sigma)

        x_fm = self.aligner.dm_to_fm(x_dm, t_dm)
        # At t=0: alpha≈0.999, sigma≈0.029, so x_dm ≈ x_0 + 0.029*eps
        # After division by (alpha+sigma)≈1.028, result is close but not exact
        error = (x_fm - self.x_0).abs().max().item()
        assert error < 0.15, f"At data end, x_FM should ≈ x_0, max error = {error}"

    def test_alignment_verification(self):
        """Verify that DM→FM mapping matches direct FM interpolant construction."""
        t_dm = torch.tensor([200.0, 600.0])
        result = self.aligner.verify_alignment(self.x_0, self.epsilon, t_dm)

        assert result["max_error"] < 1e-4, (
            f"Alignment error too high: {result['max_error']}"
        )

    def test_fm_interpolant_boundaries(self):
        """FM interpolant: x(0) = noise, x(1) = data."""
        # At t_FM = 0: x = (1-0)*noise + 0*data = noise
        t_fm_0 = torch.tensor([0.0, 0.0])
        x_at_0 = self.aligner.construct_fm_interpolant(self.epsilon, self.x_0, t_fm_0)
        assert torch.allclose(x_at_0, self.epsilon, atol=1e-6), "x_FM(0) should equal noise"

        # At t_FM = 1: x = (1-1)*noise + 1*data = data
        t_fm_1 = torch.tensor([1.0, 1.0])
        x_at_1 = self.aligner.construct_fm_interpolant(self.epsilon, self.x_0, t_fm_1)
        assert torch.allclose(x_at_1, self.x_0, atol=1e-6), "x_FM(1) should equal data"

    def test_dm_interpolant_construction(self):
        """Verify DM interpolant: x_t = α·x_0 + σ·ε."""
        t_dm = torch.tensor([500.0, 500.0])
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)

        x_dm = self.aligner.construct_dm_interpolant(self.x_0, self.epsilon, alpha, sigma)

        # Manual computation
        alpha_b = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sigma_b = sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_dm_manual = alpha_b * self.x_0 + sigma_b * self.epsilon

        assert torch.allclose(x_dm, x_dm_manual, atol=1e-6), "DM interpolant mismatch"

    def test_round_trip_fm_dm_fm(self):
        """FM → DM → FM round trip should preserve the FM interpolant."""
        t_fm = torch.tensor([0.3, 0.7])
        x_fm = self.aligner.construct_fm_interpolant(self.epsilon, self.x_0, t_fm)

        # FM → DM
        x_dm = self.aligner.fm_to_dm(x_fm, t_fm)
        # DM → FM
        t_dm = self.mapper.fm_to_dm_batched(t_fm)
        x_fm_recovered = self.aligner.dm_to_fm(x_dm, t_dm)

        error = (x_fm_recovered - x_fm).abs().max().item()
        assert error < 0.1, f"FM→DM→FM round-trip error: {error}"

    def test_multiple_timesteps(self):
        """Alignment should work for a range of timesteps."""
        for t_val in [50, 200, 500, 800, 950]:
            t_dm = torch.tensor([float(t_val), float(t_val)])
            result = self.aligner.verify_alignment(self.x_0, self.epsilon, t_dm)
            assert result["max_error"] < 1e-3, (
                f"Alignment failed at t_DM={t_val}: error = {result['max_error']}"
            )
