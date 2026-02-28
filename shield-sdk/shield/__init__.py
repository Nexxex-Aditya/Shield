"""
Shield SDK — Python client for the Shield AI Platform API.

Usage:
    pip install shield-sdk

    import shield
    client = shield.Client(api_key="sk-your-key")
    result = client.retrieve("What is our refund policy?")
"""

from .client import Client

__version__ = "0.1.0"
__all__ = ["Client"]
