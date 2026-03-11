"""
Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment

A modular implementation of the Diff2Flow framework (CVPR 2025) that bridges
diffusion and flow matching paradigms via timestep rescaling, interpolant
alignment, and velocity field derivation.
"""

__version__ = "0.1.0"

from .schedules import NoiseScheduleVP, NoiseScheduleVE
from .timestep_mapping import TimestepMapper
from .interpolant_align import InterpolantAligner
from .velocity import VelocityDeriver
from .converter import Diff2FlowConverter
