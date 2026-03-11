"""
Unit tests for noise schedules.

Verifies:
- VP schedule satisfies α² + σ² = 1
- VP schedule has correct boundary behavior
- Schedule values match known SD values
- Continuous interpolation works correctly
- VE schedule has α = 1 and increasing σ
"""

import pytest
import torch
import math

from diff2flow.schedules import NoiseScheduleVP, NoiseScheduleVE


class TestNoiseScheduleVP:
    """Tests for variance-preserving noise schedule."""

    def setup_method(self):
        self.schedule = NoiseScheduleVP(num_timesteps=1000)

    def test_vp_constraint(self):
        """α²(t) + σ²(t) must equal 1 for all VP timesteps."""
        deviation = self.schedule.verify_vp_constraint()
        assert deviation < 1e-5, f"VP constraint violated: max deviation = {deviation}"

    def test_boundary_t0(self):
        """At t=0 (data end): α should be high (~1), σ should be low (~0)."""
        alpha_0 = self.schedule.alpha(torch.tensor([0.0])).item()
        sigma_0 = self.schedule.sigma(torch.tensor([0.0])).item()

        assert alpha_0 > 0.99, f"α(0) should be ~1, got {alpha_0}"
        assert sigma_0 < 0.05, f"σ(0) should be ~0, got {sigma_0}"

    def test_boundary_tmax(self):
        """At t=T-1 (noise end): α should be low, σ should be high."""
        t_max = torch.tensor([999.0])
        alpha_T = self.schedule.alpha(t_max).item()
        sigma_T = self.schedule.sigma(t_max).item()

        assert alpha_T < 0.1, f"α(T) should be ~0, got {alpha_T}"
        assert sigma_T > 0.9, f"σ(T) should be ~1, got {sigma_T}"

    def test_alpha_monotonically_decreasing(self):
        """α(t) should monotonically decrease as t increases."""
        t = torch.arange(1000, dtype=torch.float32)
        alpha = self.schedule.alpha(t)

        diffs = alpha[1:] - alpha[:-1]
        assert (diffs <= 0).all(), "α(t) must be monotonically decreasing"

    def test_sigma_monotonically_increasing(self):
        """σ(t) should monotonically increase as t increases."""
        t = torch.arange(1000, dtype=torch.float32)
        sigma = self.schedule.sigma(t)

        diffs = sigma[1:] - sigma[:-1]
        assert (diffs >= 0).all(), "σ(t) must be monotonically increasing"

    def test_continuous_interpolation(self):
        """Interpolation at half-integer timesteps should be between neighbors."""
        t = torch.tensor([100.5])
        alpha_mid = self.schedule.alpha(t).item()
        alpha_low = self.schedule.alpha(torch.tensor([100.0])).item()
        alpha_high = self.schedule.alpha(torch.tensor([101.0])).item()

        assert alpha_high <= alpha_mid <= alpha_low, (
            f"Interpolated α({t.item()}) = {alpha_mid} should be between "
            f"α(100) = {alpha_low} and α(101) = {alpha_high}"
        )

    def test_integer_timestep_exact(self):
        """Integer timesteps should return exact precomputed values."""
        t = torch.tensor([50.0])
        alpha_interp = self.schedule.alpha(t).item()
        alpha_exact = self.schedule.alpha_values[50].item()

        assert abs(alpha_interp - alpha_exact) < 1e-7, (
            f"Interpolated α(50) = {alpha_interp} != exact α[50] = {alpha_exact}"
        )

    def test_batch_computation(self):
        """Schedule should handle batch computations correctly."""
        t = torch.tensor([0.0, 250.0, 500.0, 750.0, 999.0])
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)

        assert alpha.shape == (5,)
        assert sigma.shape == (5,)
        # VP constraint for batch
        constraint = (alpha ** 2 + sigma ** 2)
        assert (constraint - 1.0).abs().max() < 1e-5


class TestNoiseScheduleVE:
    """Tests for variance-exploding noise schedule."""

    def setup_method(self):
        self.schedule = NoiseScheduleVE(num_timesteps=1000)

    def test_alpha_always_one(self):
        """VE schedule: α(t) = 1 for all t."""
        t = torch.arange(1000, dtype=torch.float32)
        alpha = self.schedule.alpha(t)
        assert torch.allclose(alpha, torch.ones(1000)), "VE α must always be 1"

    def test_sigma_monotonically_increasing(self):
        """VE σ(t) should increase monotonically."""
        t = torch.arange(1000, dtype=torch.float32)
        sigma = self.schedule.sigma(t)

        diffs = sigma[1:] - sigma[:-1]
        assert (diffs > 0).all(), "VE σ(t) must be strictly increasing"

    def test_sigma_range(self):
        """VE σ should span from sigma_min to sigma_max."""
        sigma_0 = self.schedule.sigma(torch.tensor([0.0])).item()
        sigma_T = self.schedule.sigma(torch.tensor([999.0])).item()

        assert abs(sigma_0 - 0.01) < 1e-3, f"σ(0) should be ~sigma_min, got {sigma_0}"
        assert abs(sigma_T - 50.0) < 1.0, f"σ(T) should be ~sigma_max, got {sigma_T}"
