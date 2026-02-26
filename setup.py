"""Setup configuration for DeepArtNet."""

from setuptools import setup, find_packages

setup(
    name="deepartnet",
    version="0.1.0",
    description="Multi-attribute fine art classification using a CNN-RNN hybrid architecture",
    author="DeepArtNet Authors",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
