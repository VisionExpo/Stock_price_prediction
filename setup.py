from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="stock_price_prediction",
    version="1.0.0",
    author="Vishal Gorule",
    author_email="gorulevishal984@gmail.com",
    description="Stock price prediction using LSTM neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VisionExpo/Stock_price_prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stock-predict=src.cli:main",
        ],
    },
)
