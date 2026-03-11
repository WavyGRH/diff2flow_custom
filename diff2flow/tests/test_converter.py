"""
Integration tests for the Diff2FlowConverter.

Tests the full pipeline: prepare_training_sample, compute_loss,
and diffusion_to_velocity working together.
"""

import pytest
import torch

from diff2flow.converter import Diff2FlowConverter, TrainingSample


class TestDiff2FlowConverter:
    """Integration tests for the unified converter."""

    def setup_method(self):
        self.converter = Diff2FlowConverter(parameterization="v")

        torch.manual_seed(42)
        self.x_data = torch.randn(2, 4, 8, 8)
        self.epsilon = torch.randn(2, 4, 8, 8)

    def test_prepare_training_sample_shapes(self):
        """All output tensors should have correct shapes."""
        t_fm = torch.tensor([0.3, 0.7])
        sample = self.converter.prepare_training_sample(self.x_data, self.epsilon, t_fm)

        assert sample.x_fm.shape == (2, 4, 8, 8), f"x_fm shape: {sample.x_fm.shape}"
        assert sample.x_dm.shape == (2, 4, 8, 8), f"x_dm shape: {sample.x_dm.shape}"
        assert sample.t_dm_bar.shape == (2,), f"t_dm_bar shape: {sample.t_dm_bar.shape}"
        assert sample.t_fm.shape == (2,), f"t_fm shape: {sample.t_fm.shape}"
        assert sample.v_target.shape == (2, 4, 8, 8), f"v_target shape: {sample.v_target.shape}"

    def test_target_velocity_is_data_minus_noise(self):
        """Target velocity should be x_data - epsilon."""
        t_fm = torch.tensor([0.5, 0.5])
        sample = self.converter.prepare_training_sample(self.x_data, self.epsilon, t_fm)

        expected = self.x_data - self.epsilon
        assert torch.allclose(sample.v_target, expected, atol=1e-6), (
            "v_target should be data - noise"
        )

    def test_t_dm_bar_in_valid_range(self):
        """Continuous DM timestep should be in [0, 999]."""
        torch.manual_seed(123)
        B = 10
        t_fm = torch.rand(B)
        x_data = torch.randn(B, 4, 8, 8)
        x_noise = torch.randn(B, 4, 8, 8)
        sample = self.converter.prepare_training_sample(x_data, x_noise, t_fm)

        assert sample.t_dm_bar.min() >= 0.0, f"t_dm_bar below 0: {sample.t_dm_bar.min()}"
        assert sample.t_dm_bar.max() <= 999.0, f"t_dm_bar above 999: {sample.t_dm_bar.max()}"

    def test_loss_with_perfect_model(self):
        """Loss should be ~0 when model output matches the Diff2Flow target."""
        t_fm = torch.tensor([0.3, 0.7])
        sample = self.converter.prepare_training_sample(self.x_data, self.epsilon, t_fm)

        # Simulate perfect v-prediction model
        schedule = self.converter.schedule
        t_dm = sample.t_dm_bar
        alpha = schedule.alpha(t_dm)
        sigma = schedule.sigma(t_dm)

        while alpha.dim() < self.x_data.dim():
            alpha = alpha.unsqueeze(-1)
        while sigma.dim() < self.x_data.dim():
            sigma = sigma.unsqueeze(-1)

        # Perfect v-prediction
        perfect_v = alpha * self.epsilon - sigma * self.x_data
        loss = self.converter.compute_loss(perfect_v, sample)

        assert loss.item() < 1e-3, f"Loss with perfect model should be ~0, got {loss.item()}"

    def test_naive_fm_loss_is_different(self):
        """Naive FM loss should typically differ from Diff2Flow loss."""
        t_fm = torch.tensor([0.4, 0.6])
        sample = self.converter.prepare_training_sample(self.x_data, self.epsilon, t_fm)

        random_output = torch.randn_like(self.x_data)
        d2f_loss = self.converter.compute_loss(random_output, sample)
        naive_loss = self.converter.compute_naive_fm_loss(random_output, sample)

        # They should both be positive but generally different
        assert d2f_loss.item() > 0, "Diff2Flow loss should be positive"
        assert naive_loss.item() > 0, "Naive FM loss should be positive"

    def test_converter_different_parameterizations(self):
        """Converter should work with all parameterization types."""
        for param in ["v", "epsilon", "x0"]:
            converter = Diff2FlowConverter(parameterization=param)
            t_fm = torch.tensor([0.5, 0.5])
            sample = converter.prepare_training_sample(self.x_data, self.epsilon, t_fm)

            assert sample.x_fm.shape == (2, 4, 8, 8), (
                f"Failed for parameterization={param}"
            )

    def test_end_to_end_flow(self):
        """Full end-to-end test: prepare → model → velocity → loss."""
        t_fm = torch.tensor([0.3, 0.7])

        # Step 1: Prepare training sample
        sample = self.converter.prepare_training_sample(self.x_data, self.epsilon, t_fm)

        # Step 2: Simulate model forward pass (random output)
        model_output = torch.randn(2, 4, 8, 8)

        # Step 3: Compute Diff2Flow loss
        loss = self.converter.compute_loss(model_output, sample)

        # Step 4: Verify loss is a scalar and positive
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive for random model output"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_diffusion_to_velocity(self):
        """diffusion_to_velocity should produce correct-shaped output."""
        t_dm = torch.tensor([200.0, 600.0])
        model_output = torch.randn(2, 4, 8, 8)
        x_dm = torch.randn(2, 4, 8, 8)

        v_fm = self.converter.diffusion_to_velocity(model_output, x_dm, t_dm)

        assert v_fm.shape == (2, 4, 8, 8), f"Velocity shape: {v_fm.shape}"
        assert torch.isfinite(v_fm).all(), "Velocity should be finite"
