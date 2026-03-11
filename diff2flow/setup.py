from setuptools import setup, find_packages

setup(
    name="diff2flow",
    version="0.1.0",
    description="Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment",
    author="Implementation based on Schusterbauer, Gui, Fundel & Ommer (CompVis @ LMU Munich)",
    url="https://arxiv.org/abs/2506.02221",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "train": ["wandb>=0.16.0", "accelerate>=0.25.0", "datasets>=2.16.0"],
    },
    license="MIT",
)
