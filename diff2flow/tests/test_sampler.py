"""
Unit tests for the Euler ODE sampler.

Verifies:
- Correct trajectory from identity velocity (x + dt * v)
- NFE scheduling is correct
- Uniform timestep spacing
- Straightness metric computation
"""

import pytest
import torch

from diff2flow.sampler import EulerSampler


class TestEulerSampler:
    """Tests for the flow matching Euler sampler."""

    def test_identity_velocity(self):
        """With v = constant, result should be x_0 + v (integrated over [0,1])."""
        sampler = EulerSampler(num_steps=100)

        x_init = torch.zeros(1, 4, 8, 8)
        constant_v = torch.ones(1, 4, 8, 8)

        def model_fn(x, t, enc_states=None):
            return constant_v

        result = sampler.sample(
            model_fn=model_fn,
            x_init=x_init,
            show_progress=False,
        )

        # x_final = x_init + integral(v, 0, 1) = 0 + 1 = 1
        expected = torch.ones(1, 4, 8, 8)
        error = (result - expected).abs().max().item()
        assert error < 0.02, f"Identity velocity test: max error = {error}"

    def test_linear_velocity(self):
        """With v = target - x_init (pointing from init to target), should reach target."""
        sampler = EulerSampler(num_steps=1)  # One Euler step

        x_init = torch.zeros(1, 4, 8, 8)
        target = torch.ones(1, 4, 8, 8) * 5.0

        def model_fn(x, t, enc_states=None):
            return target - x_init  # Constant velocity pointing to target

        result = sampler.sample(
            model_fn=model_fn,
            x_init=x_init,
            show_progress=False,
        )

        # With 1 step: x_1 = x_0 + 1.0 * (target - x_0) = target
        error = (result - target).abs().max().item()
        assert error < 1e-5, f"Linear velocity test: max error = {error}"

    def test_timestep_schedule(self):
        """Timesteps should be uniformly spaced from 0 to 1."""
        sampler = EulerSampler(num_steps=10)

        assert sampler.timesteps[0].item() == 0.0, "First timestep should be 0"
        assert sampler.timesteps[-1].item() == 1.0, "Last timestep should be 1"
        assert len(sampler.timesteps) == 11, "Should have N+1 timestep boundaries"

        # Check uniform spacing
        diffs = sampler.timesteps[1:] - sampler.timesteps[:-1]
        assert torch.allclose(diffs, torch.tensor(0.1), atol=1e-6), "Timesteps should be uniform"

    def test_dt_value(self):
        """dt should equal 1/N."""
        for N in [2, 4, 10, 25, 50]:
            sampler = EulerSampler(num_steps=N)
            assert abs(sampler.dt - 1.0 / N) < 1e-10, f"dt should be 1/{N}"

    def test_trajectory_recording(self):
        """sample_with_trajectory should return N+1 states."""
        sampler = EulerSampler(num_steps=5)
        x_init = torch.randn(1, 4, 8, 8)

        def model_fn(x, t, enc_states=None):
            return torch.ones_like(x)

        trajectory = sampler.sample_with_trajectory(
            model_fn=model_fn,
            x_init=x_init,
        )

        assert len(trajectory) == 6, f"Trajectory should have 6 points, got {len(trajectory)}"
        # First point should be the initial noise
        assert torch.allclose(trajectory[0], x_init), "First trajectory point should be x_init"

    def test_straightness_perfect(self):
        """Perfectly straight trajectory should have straightness = 1."""
        sampler = EulerSampler(num_steps=10)

        # Create a perfectly straight trajectory
        trajectory = [torch.zeros(1, 4) + i * 0.1 * torch.ones(1, 4) for i in range(11)]

        straightness = sampler.compute_straightness(trajectory)
        assert abs(straightness - 1.0) < 1e-4, f"Straightness should be 1.0, got {straightness}"

    def test_straightness_curved(self):
        """Curved trajectory should have straightness < 1."""
        sampler = EulerSampler(num_steps=4)

        # Create a curved trajectory (detour)
        trajectory = [
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 1.0]]),   # Detour up
            torch.tensor([[2.0, 0.0]]),   # Back down
            torch.tensor([[3.0, 1.0]]),   # Up again
            torch.tensor([[4.0, 0.0]]),   # End
        ]

        straightness = sampler.compute_straightness(trajectory)
        assert straightness < 1.0, f"Curved trajectory straightness should be < 1, got {straightness}"

    def test_different_nfe(self):
        """Sampler should work with various NFE values."""
        for nfe in [2, 4, 10, 25, 50]:
            sampler = EulerSampler(num_steps=nfe)
            x_init = torch.randn(1, 4, 4, 4)

            def model_fn(x, t, enc_states=None):
                return torch.zeros_like(x)

            result = sampler.sample(model_fn, x_init, show_progress=False)
            # With zero velocity, result should equal input
            assert torch.allclose(result, x_init, atol=1e-6), f"Failed for NFE={nfe}"
