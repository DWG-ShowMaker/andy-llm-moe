from setuptools import setup, find_packages

setup(
    name="chinese_moe_model",
    version="0.1.0",
    description="A small Chinese model training project using MoE architecture.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 