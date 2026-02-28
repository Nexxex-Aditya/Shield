"""
AgentVault — Provider Factory
Auto-detect backend from URI, instantiate the right provider.
Customers can register custom providers at runtime.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Type
from urllib.parse import urlparse

from .base import DatabaseProvider

logger = logging.getLogger("agentvault.database.factory")

# ---------------------------------------------------------------------------
# Built-in scheme → provider mapping (lazy imports to avoid hard deps)
# ---------------------------------------------------------------------------
_BUILTIN_SCHEMES: dict[str, str] = {
    "sqlite":     "server.database.providers.sqlite_provider.SQLiteProvider",
    "postgresql": "server.database.providers.postgres_provider.PostgresProvider",
    "postgres":   "server.database.providers.postgres_provider.PostgresProvider",
    "mongodb":    "server.database.providers.mongo_provider.MongoProvider",
    "mongodb+srv":"server.database.providers.mongo_provider.MongoProvider",
    "s3":         "server.database.providers.s3_provider.S3Provider",
    "oracle":     "server.database.providers.oracle_provider.OracleProvider",
    "oracle+tcp": "server.database.providers.oracle_provider.OracleProvider",
}

# Custom provider registry
_CUSTOM_REGISTRY: dict[str, Type[DatabaseProvider]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_provider(scheme: str, provider_class: Type[DatabaseProvider]) -> None:
    """
    Register a custom database provider for a URI scheme.

    Example::

        from server.database import register_provider
        from my_company.datalake import DataLakeProvider

        register_provider("datalake", DataLakeProvider)
        # Then set AGENTVAULT_DB=datalake://host/bucket
    """
    if not issubclass(provider_class, DatabaseProvider):
        raise TypeError(f"{provider_class} must subclass DatabaseProvider")
    _CUSTOM_REGISTRY[scheme.lower()] = provider_class
    logger.info("Registered custom provider '%s' → %s", scheme, provider_class.__name__)


def parse_uri(uri: str) -> tuple[str, str]:
    """
    Parse a connection URI and return (scheme, cleaned_uri).

    Handles edge cases:
    - Plain file paths → treated as sqlite
    - ``sqlite:///path`` → sqlite
    - ``postgresql://...`` → postgresql
    - ``s3://bucket/prefix`` → s3
    """
    uri = uri.strip()

    # Bare file path (no scheme) → sqlite
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", uri):
        return "sqlite", uri

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    return scheme, uri


def _import_class(dotted_path: str) -> Type[DatabaseProvider]:
    """Dynamic import of a provider class from a dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


async def create_provider(uri: str) -> DatabaseProvider:
    """
    Create and connect the right provider for the given URI.

    Steps:
    1. Parse URI → extract scheme
    2. Check custom registry, then built-in schemes
    3. Instantiate provider
    4. Call ``connect()`` to initialise schema / pool
    5. Run ``health_check()`` to confirm connectivity
    6. Return ready-to-use provider
    """
    scheme, cleaned = parse_uri(uri)
    logger.info("Resolving provider for scheme '%s'", scheme)

    # 1. Custom registry first
    if scheme in _CUSTOM_REGISTRY:
        provider_class = _CUSTOM_REGISTRY[scheme]
        logger.info("Using custom provider: %s", provider_class.__name__)
    elif scheme in _BUILTIN_SCHEMES:
        provider_class = _import_class(_BUILTIN_SCHEMES[scheme])
        logger.info("Using built-in provider: %s", provider_class.__name__)
    else:
        raise ValueError(
            f"Unsupported database scheme '{scheme}'. "
            f"Supported: {', '.join(sorted(set(list(_BUILTIN_SCHEMES) + list(_CUSTOM_REGISTRY))))}"
        )

    # 2. Instantiate
    provider = provider_class(cleaned)

    # 3. Connect (creates schema / pool)
    await provider.connect()

    # 4. Health check
    health = await provider.health_check()
    if not health.get("ok"):
        logger.warning("Health check returned not-ok: %s", health)
    else:
        logger.info(
            "Provider ready — backend=%s latency=%.1fms",
            health.get("backend", scheme),
            health.get("latency_ms", 0),
        )

    return provider
