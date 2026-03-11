"""
Unit tests for velocity derivation.

Verifies:
- Velocity from perfect v-prediction equals data - noise
- Velocity from perfect epsilon-prediction equals data - noise
- Self-consistency: different derivation methods agree
- FM loss computation is correct
"""

import pytest
import torch

from diff2flow.schedules import NoiseScheduleVP
from diff2flow.velocity import VelocityDeriver, Parameterization


class TestVelocityDeriver:
    """Tests for FM velocity derivation from DM predictions."""

    def setup_method(self):
        self.schedule = NoiseScheduleVP(num_timesteps=1000)
        self.v_deriver = VelocityDeriver(self.schedule, Parameterization.V_PREDICTION)
        self.eps_deriver = VelocityDeriver(self.schedule, Parameterization.EPSILON)
        self.x0_deriver = VelocityDeriver(self.schedule, Parameterization.X0)

        torch.manual_seed(42)
        self.x_0 = torch.randn(2, 4, 8, 8)  # Clean data
        self.epsilon = torch.randn(2, 4, 8, 8)  # Noise

    def _construct_dm_interpolant(self, t_dm: torch.Tensor):
        """Helper: construct x_DM and related quantities."""
        alpha = self.schedule.alpha(t_dm)
        sigma = self.schedule.sigma(t_dm)

        while alpha.dim() < self.x_0.dim():
            alpha = alpha.unsqueeze(-1)
        while sigma.dim() < self.x_0.dim():
            sigma = sigma.unsqueeze(-1)

        x_dm = alpha * self.x_0 + sigma * self.epsilon
        return x_dm, alpha, sigma

    def test_velocity_from_perfect_v_prediction(self):
        """If model perfectly predicts v, velocity should equal x_0 - epsilon."""
        t_dm = torch.tensor([200.0, 600.0])
        x_dm, alpha, sigma = self._construct_dm_interpolant(t_dm)

        # Perfect v-prediction: v = alpha * epsilon - sigma * x_0
        v_true = alpha * self.epsilon - sigma * self.x_0

        v_fm = self.v_deriver.derive_velocity(v_true, x_dm, t_dm)

        # FM velocity should be: data - noise = x_0 - epsilon
        v_expected = self.x_0 - self.epsilon

        error = (v_fm - v_expected).abs().max().item()
        assert error < 1e-4, (
            f"Velocity from perfect v-pred should equal x_0 - eps, "
            f"max error = {error}"
        )

    def test_velocity_from_perfect_eps_prediction(self):
        """If model perfectly predicts epsilon, velocity should equal x_0 - eps."""
        t_dm = torch.tensor([200.0, 600.0])
        x_dm, alpha, sigma = self._construct_dm_interpolant(t_dm)

        v_fm = self.eps_deriver.derive_velocity(self.epsilon, x_dm, t_dm)

        v_expected = self.x_0 - self.epsilon
        error = (v_fm - v_expected).abs().max().item()
        assert error < 1e-4, (
            f"Velocity from perfect eps-pred should equal x_0 - eps, "
            f"max error = {error}"
        )

    def test_velocity_from_perfect_x0_prediction(self):
        """If model perfectly predicts x0, velocity should equal x_0 - eps."""
        t_dm = torch.tensor([200.0, 600.0])
        x_dm, alpha, sigma = self._construct_dm_interpolant(t_dm)

        v_fm = self.x0_deriver.derive_velocity(self.x_0, x_dm, t_dm)

        v_expected = self.x_0 - self.epsilon
        error = (v_fm - v_expected).abs().max().item()
        assert error < 1e-4, (
            f"Velocity from perfect x0-pred should equal x_0 - eps, "
            f"max error = {error}"
        )

    def test_parameterization_consistency(self):
        """All parameterizations should give the same velocity for perfect predictions."""
        t_dm = torch.tensor([300.0, 700.0])
        x_dm, alpha, sigma = self._construct_dm_interpolant(t_dm)

        # Perfect predictions for each parameterization
        v_true = alpha * self.epsilon - sigma * self.x_0

        v_from_v = self.v_deriver.derive_velocity(v_true, x_dm, t_dm)
        v_from_eps = self.eps_deriver.derive_velocity(self.epsilon, x_dm, t_dm)
        v_from_x0 = self.x0_deriver.derive_velocity(self.x_0, x_dm, t_dm)

        assert torch.allclose(v_from_v, v_from_eps, atol=1e-4), "v-pred and eps-pred disagree"
        assert torch.allclose(v_from_v, v_from_x0, atol=1e-4), "v-pred and x0-pred disagree"

    def test_fm_loss_zero_for_perfect_velocity(self):
        """FM loss should be zero when predicted velocity matches target."""
        v_target = self.x_0 - self.epsilon
        loss = self.v_deriver.compute_fm_loss(v_target, self.x_0, self.epsilon)
        assert loss.item() < 1e-10, f"Loss should be ~0, got {loss.item()}"

    def test_fm_loss_positive_for_wrong_velocity(self):
        """FM loss should be positive when prediction doesn't match."""
        wrong_velocity = torch.randn_like(self.x_0)
        loss = self.v_deriver.compute_fm_loss(wrong_velocity, self.x_0, self.epsilon)
        assert loss.item() > 0, "Loss should be positive for wrong prediction"

    def test_velocity_shape_preservation(self):
        """Output velocity should have same shape as input."""
        t_dm = torch.tensor([400.0, 400.0])
        x_dm, alpha, sigma = self._construct_dm_interpolant(t_dm)
        v_pred = torch.randn_like(x_dm)

        v_fm = self.v_deriver.derive_velocity(v_pred, x_dm, t_dm)
        assert v_fm.shape == x_dm.shape, f"Shape mismatch: {v_fm.shape} != {x_dm.shape}"

    def test_multiple_timesteps(self):
        """Velocity derivation should work across different timesteps."""
        for t_val in [10, 100, 500, 900, 990]:
            t_dm = torch.tensor([float(t_val), float(t_val)])
            x_dm, alpha, sigma = self._construct_dm_interpolant(t_dm)
            v_true = alpha * self.epsilon - sigma * self.x_0

            v_fm = self.v_deriver.derive_velocity(v_true, x_dm, t_dm)
            v_expected = self.x_0 - self.epsilon

            error = (v_fm - v_expected).abs().max().item()
            assert error < 1e-3, f"Velocity error at t={t_val}: {error}"
