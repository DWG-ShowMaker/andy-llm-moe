from setuptools import setup, find_packages

setup(
    name="andy-llm-moe",
    version="0.1.0",
    description="A small Chinese model training project using MoE architecture.",
    author="BossAndy",
    author_email="746144374@qq.com",
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