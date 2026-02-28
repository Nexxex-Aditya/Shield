from setuptools import setup, find_packages

setup(
    name="shield-sdk",
    version="0.1.0",
    description="Python SDK for the Shield AI Platform API",
    author="Nexxex Technologies",
    author_email="sdk@nexxex.com",
    url="https://shield.nexxex.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
