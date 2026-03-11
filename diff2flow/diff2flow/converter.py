"""
Unified Diff2Flow Converter.

Orchestrates the full Diff2Flow pipeline: timestep mapping, interpolant
alignment, and velocity derivation. This is the main entry point for
converting between diffusion and flow matching paradigms.

Usage:
    converter = Diff2FlowConverter()

    # During training: prepare FM training samples from data + noise
    sample = converter.prepare_training_sample(x_data, epsilon, t_fm)

    # During inference: convert DM model output to FM velocity
    v_fm = converter.diffusion_to_velocity(model_output, x_dm, t_dm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .schedules import NoiseScheduleVP
from .timestep_mapping import TimestepMapper
from .interpolant_align import InterpolantAligner
from .velocity import VelocityDeriver, Parameterization


@dataclass
class TrainingSample:
    """Container for a prepared Diff2Flow training sample.

    Attributes:
        x_fm: FM interpolant to feed the model, shape [B, C, H, W].
        x_dm: Equivalent DM interpolant, shape [B, C, H, W].
        t_dm_bar: Continuous DM timestep for the model, shape [B].
        t_fm: FM timestep used, shape [B].
        v_target: Target FM velocity (x_data - x_noise), shape [B, C, H, W].
        x_data: Clean data sample (used as reference), shape [B, C, H, W].
        x_noise: Noise sample (used as reference), shape [B, C, H, W].
    """
    x_fm: torch.Tensor
    x_dm: torch.Tensor
    t_dm_bar: torch.Tensor
    t_fm: torch.Tensor
    v_target: torch.Tensor
    x_data: torch.Tensor
    x_noise: torch.Tensor


class Diff2FlowConverter:
    """Main Diff2Flow conversion pipeline.

    Combines timestep mapping, interpolant alignment, and velocity
    derivation into a unified interface for training and inference.

    The key innovation is that instead of naively applying FM loss to a
    diffusion model (which forces the model to learn a completely new
    parameterization), Diff2Flow:
    1. Maps FM timesteps to DM timesteps (so the model sees familiar inputs)
    2. Transforms FM interpolants to DM space (so the model processes familiar data)
    3. Derives FM velocity from DM predictions (so the output aligns with FM objectives)

    Args:
        schedule: Noise schedule (defaults to SD's VP schedule).
        parameterization: What the DM model predicts ('v', 'epsilon', 'x0').
        num_timesteps: Number of discrete DM timesteps (1000 for SD).
    """

    def __init__(
        self,
        schedule: Optional[NoiseScheduleVP] = None,
        parameterization: str = "v",
        num_timesteps: int = 1000,
    ):
        if schedule is None:
            schedule = NoiseScheduleVP(num_timesteps=num_timesteps)
        self.schedule = schedule

        # Parse parameterization
        param_map = {
            "v": Parameterization.V_PREDICTION,
            "epsilon": Parameterization.EPSILON,
            "eps": Parameterization.EPSILON,
            "x0": Parameterization.X0,
        }
        self.parameterization = param_map.get(parameterization, Parameterization.V_PREDICTION)

        # Initialize sub-modules
        self.mapper = TimestepMapper(schedule)
        self.aligner = InterpolantAligner(schedule, self.mapper)
        self.deriver = VelocityDeriver(schedule, self.parameterization)

    def prepare_training_sample(
        self,
        x_data: torch.Tensor,
        epsilon: torch.Tensor,
        t_fm: torch.Tensor,
    ) -> TrainingSample:
        """Prepare a training sample for Diff2Flow finetuning.

        Given clean data and noise, constructs:
        1. The FM interpolant x_FM at timestep t_FM
        2. The equivalent DM interpolant x_DM (via fx_inverse)
        3. The continuous DM timestep t_DM_bar (via ft_inverse)
        4. The target FM velocity (x_data - epsilon)

        The model receives (x_DM, t_DM_bar) as input and is trained
        to predict the FM velocity.

        Args:
            x_data: Clean data samples, shape [B, C, H, W].
            epsilon: Gaussian noise, shape [B, C, H, W].
            t_fm: FM timesteps, shape [B], values in [0, 1].

        Returns:
            TrainingSample with all required tensors.
        """
        # 1. Construct FM interpolant: x_FM = (1 - t) * noise + t * data
        x_fm = self.aligner.construct_fm_interpolant(epsilon, x_data, t_fm)

        # 2. Map FM timestep to continuous DM timestep
        t_dm_bar = self.mapper.fm_to_dm_batched(t_fm)

        # 3. Transform FM interpolant to DM space
        x_dm = self.aligner.fm_to_dm(x_fm, t_fm)

        # 4. Target velocity: FM velocity = data - noise
        v_target = x_data - epsilon

        return TrainingSample(
            x_fm=x_fm,
            x_dm=x_dm,
            t_dm_bar=t_dm_bar,
            t_fm=t_fm,
            v_target=v_target,
            x_data=x_data,
            x_noise=epsilon,
        )

    def diffusion_to_velocity(
        self,
        model_output: torch.Tensor,
        x_dm: torch.Tensor,
        t_dm: torch.Tensor,
    ) -> torch.Tensor:
        """Convert diffusion model output to FM velocity.

        Takes the raw output of the diffusion model (v-pred, eps, or x0)
        and derives the corresponding FM velocity field.

        Args:
            model_output: Raw DM model output, shape [B, C, H, W].
            x_dm: DM interpolant (model input), shape [B, C, H, W].
            t_dm: DM timestep(s), shape [B].

        Returns:
            FM velocity, shape [B, C, H, W].
        """
        return self.deriver.derive_velocity(model_output, x_dm, t_dm)

    def compute_loss(
        self,
        model_output: torch.Tensor,
        sample: TrainingSample,
    ) -> torch.Tensor:
        """Compute the Diff2Flow training loss.

        The model receives (x_DM, t_DM_bar) and predicts in its native
        parameterization. We derive the FM velocity from this prediction
        and compare against the target velocity.

        L = ||derive_velocity(model_output, x_DM, t_DM) - v_target||²

        Args:
            model_output: Model's raw output in its native parameterization.
            sample: TrainingSample from prepare_training_sample().

        Returns:
            Scalar loss value.
        """
        # Derive FM velocity from the model's prediction
        v_pred = self.diffusion_to_velocity(
            model_output, sample.x_dm, sample.t_dm_bar
        )

        # MSE loss against target FM velocity
        loss = torch.mean((v_pred - sample.v_target) ** 2)
        return loss

    def compute_naive_fm_loss(
        self,
        model_velocity: torch.Tensor,
        sample: TrainingSample,
    ) -> torch.Tensor:
        """Compute naive FM loss (baseline for comparison).

        This directly uses the model output as velocity prediction
        WITHOUT the Diff2Flow objective alignment. The paper shows
        this performs significantly worse, especially with LoRA.

        L = ||model_output - v_target||²

        Args:
            model_velocity: Model output treated directly as velocity.
            sample: TrainingSample from prepare_training_sample().

        Returns:
            Scalar loss value.
        """
        loss = torch.mean((model_velocity - sample.v_target) ** 2)
        return loss
