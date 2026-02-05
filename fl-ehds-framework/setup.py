"""FL-EHDS Framework Setup."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fl-ehds",
    version="1.0.0",
    author="Fabio Liberti",
    author_email="fabio.liberti@unimercatorum.it",
    description="Privacy-Preserving Federated Learning Framework for EHDS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/fl-ehds-framework",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "torch>=2.0.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "cryptography>=41.0.0",
        "structlog>=24.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "mypy>=1.8.0",
        ],
        "healthcare": [
            "fhir.resources>=7.0.0",
            "hl7apy>=1.3.4",
        ],
        "privacy": [
            "opacus>=1.4.0",
        ],
    },
)
