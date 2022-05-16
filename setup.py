from setuptools import setup


setup(
    name="sngp-torch",
    version="0.1",
    description="Simple library implementing Spectral-normalized Neural Gaussian Processes (SNGP) in PyTorch.",
    install_requires=[
        "torch",
    ],
    python_requires=">=3.9",
    extras_require={
        'test': ['pytest', 'sklearn'],
    }
)