from setuptools import setup, find_packages
import platform
import os

# Common dependencies for all platforms
common_requires = [
    "snac>=0.1.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "numpy>=1.20.0",
    "accelerate>=0.20.0",
]

# Define platform-specific dependencies
install_requires = common_requires.copy()

# Only add vllm on non-Windows platforms
if platform.system() != "Windows":
    try:
        # Try to add vllm with a specific version if available
        install_requires.append("vllm>=0.1.4")
    except Exception:
        # If vllm is not available, just continue without it
        print("Warning: vllm package not available for this platform. Will use transformers instead.")

# Read the README file for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="orpheus-speech",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    author="Amu Varma",
    author_email="amu@canopylabs.com",
    description="Orpheus Text-to-Speech System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/canopyai/orpheus-tts",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",  # Updated to 3.8 for better compatibility with dependencies
    keywords="text-to-speech, tts, ai, speech synthesis, orpheus",
)