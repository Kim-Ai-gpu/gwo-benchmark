from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="gwo-benchmark",
    version="0.2.10",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "pandas",
    ],
    author="Youngseong Kim",
    author_email="dafaafafaf33@gmail.com",
    description="A benchmark for Generalized Windowed Operations in neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kim-Ai-gpu/gwo-benchmark",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
