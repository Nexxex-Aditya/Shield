"""
AgentVault — Universal Database Package
Pluggable provider architecture for any backend.

Usage::

    from server.database import create_store, DatabaseStore

    # Auto-detect from URI
    store = await create_store("postgresql://user:pass@host/db")

    # Or old-style (defaults to SQLite)
    store = DatabaseStore("agentvault.db")
    await store.connect()
"""

from .base import DatabaseProvider
from .factory import create_provider, register_provider, parse_uri
from .store import DatabaseStore


async def create_store(uri: str = "agentvault.db") -> DatabaseStore:
    """
    One-liner to create a fully connected ``DatabaseStore``.

    Parses the URI, picks the right provider, connects, health-checks,
    and returns a ready-to-use store.
    """
    store = DatabaseStore(uri)
    await store.connect()
    return store


__all__ = [
    "DatabaseProvider",
    "DatabaseStore",
    "create_store",
    "create_provider",
    "register_provider",
    "parse_uri",
]
