"""
AgentVault — Tool / Service Registry

Unified catalog of connectors with guided setup, auto-configuration,
health monitoring, and smart suggestions. Makes Shield the place where
you plug in every tool, service, and database in your autonomous stack.

Design:
    - ConnectorSpec defines WHAT can be connected (blueprint)
    - ConnectorStatus tracks LIVE connections and their health
    - Built-in connector library covers common services
    - Users add custom connectors via API or dashboard
    - Health checks run periodically to track uptime/latency
    - Feeds data into Surveillance + CIBIL modules
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

import yaml
import httpx

from .models import (
    ConnectorCategory,
    ConnectorSpec,
    ConnectorStatus,
    MCPServerConfig,
)

logger = logging.getLogger("agentvault.registry")

# ---------------------------------------------------------------------------
# Built-in Connector Library
# ---------------------------------------------------------------------------

BUILTIN_CONNECTORS: list[dict[str, Any]] = [
    {
        "connector_id": "postgresql",
        "name": "PostgreSQL",
        "category": "database",
        "description": "Open-source relational database. Industry standard for transactional workloads.",
        "icon": "🐘",
        "config_schema": {
            "host": {"type": "string", "default": "localhost", "required": True},
            "port": {"type": "integer", "default": 5432},
            "database": {"type": "string", "required": True},
            "username": {"type": "string", "required": True},
            "password": {"type": "string", "required": True, "secret": True},
        },
        "setup_steps": [
            "Enter your PostgreSQL host address and port",
            "Provide database name and credentials",
            "Shield will test the connection and begin monitoring",
        ],
        "default_port": 5432,
        "tags": ["sql", "relational", "transactional"],
    },
    {
        "connector_id": "mysql",
        "name": "MySQL",
        "category": "database",
        "description": "Popular open-source relational database used in web applications.",
        "icon": "🐬",
        "config_schema": {
            "host": {"type": "string", "default": "localhost", "required": True},
            "port": {"type": "integer", "default": 3306},
            "database": {"type": "string", "required": True},
            "username": {"type": "string", "required": True},
            "password": {"type": "string", "required": True, "secret": True},
        },
        "setup_steps": [
            "Enter your MySQL host and port",
            "Provide database name and credentials",
            "Shield will verify and start monitoring queries",
        ],
        "default_port": 3306,
        "tags": ["sql", "relational", "web"],
    },
    {
        "connector_id": "redis",
        "name": "Redis",
        "category": "database",
        "description": "In-memory data store for caching, sessions, and real-time analytics.",
        "icon": "⚡",
        "config_schema": {
            "host": {"type": "string", "default": "localhost", "required": True},
            "port": {"type": "integer", "default": 6379},
            "password": {"type": "string", "secret": True},
            "db": {"type": "integer", "default": 0},
        },
        "setup_steps": [
            "Enter your Redis host and port",
            "Optionally provide a password and DB index",
            "Shield will ping the server and begin monitoring",
        ],
        "default_port": 6379,
        "tags": ["cache", "key-value", "real-time"],
    },
    {
        "connector_id": "slack",
        "name": "Slack",
        "category": "messaging",
        "description": "Team messaging platform. Agents can send notifications and receive commands.",
        "icon": "💬",
        "config_schema": {
            "bot_token": {"type": "string", "required": True, "secret": True},
            "default_channel": {"type": "string", "default": "#general"},
            "webhook_url": {"type": "string"},
        },
        "setup_steps": [
            "Create a Slack Bot at api.slack.com/apps",
            "Copy the Bot OAuth Token (xoxb-...)",
            "Shield will verify the token and list accessible channels",
        ],
        "tags": ["chat", "notifications", "collaboration"],
    },
    {
        "connector_id": "discord",
        "name": "Discord",
        "category": "messaging",
        "description": "Community platform. Agents can interact with servers, channels, and users.",
        "icon": "🎮",
        "config_schema": {
            "bot_token": {"type": "string", "required": True, "secret": True},
            "guild_id": {"type": "string"},
            "default_channel_id": {"type": "string"},
        },
        "setup_steps": [
            "Create a Discord Bot at discord.com/developers",
            "Copy the Bot Token from the Bot settings page",
            "Shield will connect and list accessible servers",
        ],
        "tags": ["chat", "community", "bots"],
    },
    {
        "connector_id": "github",
        "name": "GitHub",
        "category": "dev_tools",
        "description": "Code hosting platform. Agents can read repos, create issues, and manage PRs.",
        "icon": "🐙",
        "config_schema": {
            "token": {"type": "string", "required": True, "secret": True},
            "org": {"type": "string"},
            "default_repo": {"type": "string"},
        },
        "setup_steps": [
            "Generate a Personal Access Token at github.com/settings/tokens",
            "Select required scopes (repo, issues, pull_requests)",
            "Shield will verify access and list available repositories",
        ],
        "tags": ["git", "code", "ci-cd", "issues"],
    },
    {
        "connector_id": "s3",
        "name": "Amazon S3",
        "category": "storage",
        "description": "Cloud object storage. Store and retrieve files, backups, and data exports.",
        "icon": "📦",
        "config_schema": {
            "access_key": {"type": "string", "required": True, "secret": True},
            "secret_key": {"type": "string", "required": True, "secret": True},
            "region": {"type": "string", "default": "us-east-1"},
            "bucket": {"type": "string", "required": True},
        },
        "setup_steps": [
            "Create IAM credentials with S3 access in AWS Console",
            "Enter Access Key, Secret Key, and target bucket",
            "Shield will verify bucket access and start monitoring uploads/downloads",
        ],
        "tags": ["aws", "files", "backup", "cloud-storage"],
    },
    {
        "connector_id": "rest_api",
        "name": "REST API (Generic)",
        "category": "api",
        "description": "Connect to any REST API. Define base URL, auth headers, and endpoints.",
        "icon": "🌐",
        "config_schema": {
            "base_url": {"type": "string", "required": True},
            "auth_type": {"type": "string", "default": "bearer", "enum": ["none", "bearer", "api_key", "basic"]},
            "auth_value": {"type": "string", "secret": True},
            "headers": {"type": "object"},
        },
        "setup_steps": [
            "Enter the API base URL (e.g., https://api.example.com/v1)",
            "Choose auth type and provide credentials",
            "Shield will test connectivity and begin monitoring all requests",
        ],
        "tags": ["http", "webhooks", "integration"],
    },
    {
        "connector_id": "filesystem",
        "name": "Local File System",
        "category": "storage",
        "description": "Monitor and secure file operations in specified directories.",
        "icon": "📁",
        "config_schema": {
            "base_path": {"type": "string", "required": True},
            "read_only": {"type": "boolean", "default": False},
            "allowed_extensions": {"type": "array", "default": []},
        },
        "setup_steps": [
            "Specify the base directory to monitor",
            "Choose read-only or read-write mode",
            "Shield will validate the path and enforce access controls",
        ],
        "tags": ["files", "local", "disk"],
    },
    {
        "connector_id": "webhook",
        "name": "Webhook (Inbound)",
        "category": "api",
        "description": "Receive HTTP webhooks from external services. Shield validates and routes them.",
        "icon": "🔔",
        "config_schema": {
            "path": {"type": "string", "default": "/webhooks/incoming"},
            "secret": {"type": "string", "secret": True},
            "allowed_ips": {"type": "array", "default": []},
        },
        "setup_steps": [
            "Choose a webhook endpoint path",
            "Optionally set a signing secret for verification",
            "Shield will expose the endpoint and validate incoming requests",
        ],
        "tags": ["events", "inbound", "triggers"],
    },
    {
        "connector_id": "sqlite",
        "name": "SQLite",
        "category": "database",
        "description": "Lightweight file-based SQL database. Great for local development and small apps.",
        "icon": "📋",
        "config_schema": {
            "db_path": {"type": "string", "required": True},
            "read_only": {"type": "boolean", "default": False},
        },
        "setup_steps": [
            "Specify the path to your SQLite database file",
            "Choose read-only or read-write mode",
            "Shield will open the database and start monitoring queries",
        ],
        "default_port": None,
        "tags": ["sql", "local", "embedded"],
    },
]


class ServiceRegistry:
    """
    Central registry for tool/service connectors.

    Responsibilities:
    - Maintains a catalog of available connector blueprints
    - Manages live connections and their health
    - Provides guided setup flows for each connector type
    - Generates MCPServerConfig for gateway integration
    - Monitors connection health and reports anomalies
    - Suggests connectors based on usage patterns
    """

    def __init__(self, connectors_dir: str = "connectors") -> None:
        self._connectors_dir = Path(connectors_dir)
        self._catalog: dict[str, ConnectorSpec] = {}
        self._connections: dict[str, ConnectorStatus] = {}
        self._health_history: dict[str, list[dict]] = {}

    # ── Lifecycle ────────────────────────────────────────────────────

    def load_builtins(self) -> int:
        """Load all built-in connector definitions."""
        loaded = 0
        for defn in BUILTIN_CONNECTORS:
            try:
                spec = ConnectorSpec(**defn)
                self._catalog[spec.connector_id] = spec
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load builtin connector %s: %s",
                               defn.get("connector_id", "?"), e)
        logger.info("Loaded %d built-in connectors", loaded)
        return loaded

    def load_custom(self) -> int:
        """Load custom connector YAML files from connectors directory."""
        if not self._connectors_dir.exists():
            self._connectors_dir.mkdir(parents=True, exist_ok=True)
            return 0

        loaded = 0
        for fp in sorted(self._connectors_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(fp.read_text(encoding="utf-8"))
                spec = ConnectorSpec(**data)
                self._catalog[spec.connector_id] = spec
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load connector %s: %s", fp.name, e)

        if loaded:
            logger.info("Loaded %d custom connectors from %s", loaded, self._connectors_dir)
        return loaded

    # ── Catalog Operations ───────────────────────────────────────────

    def list_connectors(
        self, category: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List available connector specs, optionally filtered by category."""
        results = []
        for spec in self._catalog.values():
            if category and spec.category.value != category:
                continue
            results.append({
                "connector_id": spec.connector_id,
                "name": spec.name,
                "category": spec.category.value,
                "description": spec.description,
                "icon": spec.icon,
                "tags": spec.tags,
                "config_fields": list(spec.config_schema.keys()),
                "setup_steps": spec.setup_steps,
                "connected": any(
                    c.connector_id == spec.connector_id and c.connected
                    for c in self._connections.values()
                ),
            })
        return results

    def get_connector_spec(self, connector_id: str) -> Optional[ConnectorSpec]:
        """Get a connector blueprint by ID."""
        return self._catalog.get(connector_id)

    def register_custom(self, data: dict[str, Any]) -> ConnectorSpec:
        """Register a custom connector from raw data."""
        spec = ConnectorSpec(**data)
        if spec.connector_id in self._catalog:
            raise ValueError(f"Connector '{spec.connector_id}' already exists")
        self._catalog[spec.connector_id] = spec

        # Persist to disk
        fp = self._connectors_dir / f"{spec.connector_id}.yaml"
        fp.write_text(
            yaml.dump(spec.model_dump(mode="json"), default_flow_style=False),
            encoding="utf-8",
        )
        logger.info("Registered custom connector: %s", spec.connector_id)
        return spec

    # ── Connection Management ────────────────────────────────────────

    def connect(
        self,
        connector_id: str,
        config: dict[str, Any],
        instance_name: Optional[str] = None,
    ) -> ConnectorStatus:
        """
        Establish a connection to a service using a connector spec.
        Returns the live ConnectorStatus.
        """
        spec = self._catalog.get(connector_id)
        if not spec:
            raise ValueError(f"Unknown connector: '{connector_id}'")

        # Validate required config fields
        for field_name, field_def in spec.config_schema.items():
            if isinstance(field_def, dict) and field_def.get("required") and field_name not in config:
                raise ValueError(f"Missing required config field: '{field_name}'")

        # Create connection status
        status = ConnectorStatus(
            connector_id=connector_id,
            name=instance_name or spec.name,
            category=spec.category,
            connected=True,
            healthy=True,
            config=self._redact_secrets(config, spec.config_schema),
            connected_at=datetime.utcnow(),
            last_health_check=datetime.utcnow(),
        )

        self._connections[status.instance_id] = status
        self._health_history[status.instance_id] = []

        logger.info(
            "Connected: %s (%s) → instance %s",
            spec.name, connector_id, status.instance_id[:8],
        )
        return status

    def disconnect(self, instance_id: str) -> bool:
        """Disconnect a service instance."""
        if instance_id not in self._connections:
            return False
        status = self._connections.pop(instance_id)
        self._health_history.pop(instance_id, None)
        logger.info("Disconnected: %s (%s)", status.name, instance_id[:8])
        return True

    def list_connections(self) -> list[dict[str, Any]]:
        """List all active connections with their status."""
        return [
            {
                "instance_id": s.instance_id,
                "connector_id": s.connector_id,
                "name": s.name,
                "category": s.category.value,
                "connected": s.connected,
                "healthy": s.healthy,
                "latency_ms": round(s.latency_ms, 1),
                "total_calls": s.total_calls,
                "error_count": s.error_count,
                "uptime_percent": round(s.uptime_percent, 1),
                "connected_at": s.connected_at.isoformat() if s.connected_at else None,
                "last_health_check": s.last_health_check.isoformat() if s.last_health_check else None,
            }
            for s in self._connections.values()
        ]

    def get_connection(self, instance_id: str) -> Optional[ConnectorStatus]:
        """Get a specific connection by instance ID."""
        return self._connections.get(instance_id)

    # ── Health Monitoring ────────────────────────────────────────────

    async def health_check(self, instance_id: str) -> dict[str, Any]:
        """
        Run a health check on a connected service.
        Updates the ConnectorStatus and records history.
        """
        status = self._connections.get(instance_id)
        if not status:
            return {"error": f"Unknown instance: {instance_id}"}

        spec = self._catalog.get(status.connector_id)
        start = time.monotonic()

        try:
            # For services with HTTP health endpoints
            if spec and spec.health_check_endpoint:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(spec.health_check_endpoint)
                    healthy = resp.status_code < 400
            else:
                # Default: mark as healthy if connected
                healthy = status.connected

            latency = (time.monotonic() - start) * 1000

            status.healthy = healthy
            status.latency_ms = latency
            status.last_health_check = datetime.utcnow()

            # Update uptime tracking
            history = self._health_history.get(instance_id, [])
            history.append({
                "healthy": healthy,
                "latency_ms": round(latency, 1),
                "timestamp": datetime.utcnow().isoformat(),
            })
            # Keep last 100 checks
            if len(history) > 100:
                history = history[-100:]
            self._health_history[instance_id] = history

            # Compute uptime
            if history:
                healthy_count = sum(1 for h in history if h["healthy"])
                status.uptime_percent = (healthy_count / len(history)) * 100

            return {
                "instance_id": instance_id,
                "healthy": healthy,
                "latency_ms": round(latency, 1),
                "uptime_percent": round(status.uptime_percent, 1),
            }

        except Exception as e:
            status.healthy = False
            status.error_count += 1
            status.last_health_check = datetime.utcnow()
            logger.warning("Health check failed for %s: %s", instance_id[:8], e)
            return {
                "instance_id": instance_id,
                "healthy": False,
                "error": str(e),
            }

    async def health_check_all(self) -> list[dict[str, Any]]:
        """Run health checks on all connected services."""
        tasks = [
            self.health_check(iid) for iid in self._connections
        ]
        return await asyncio.gather(*tasks)

    def record_call(self, instance_id: str, success: bool, latency_ms: float) -> None:
        """Record a tool call for tracking stats."""
        status = self._connections.get(instance_id)
        if not status:
            return
        status.total_calls += 1
        if not success:
            status.error_count += 1
        # Running average latency
        if status.total_calls == 1:
            status.latency_ms = latency_ms
        else:
            status.latency_ms = (
                status.latency_ms * 0.9 + latency_ms * 0.1
            )

    # ── MCPGateway Integration ───────────────────────────────────────

    def to_mcp_config(self, instance_id: str) -> Optional[MCPServerConfig]:
        """
        Generate an MCPServerConfig from a connection, allowing
        the gateway to proxy tool calls to this service.
        """
        status = self._connections.get(instance_id)
        if not status:
            return None

        spec = self._catalog.get(status.connector_id)
        if not spec:
            return None

        # Build a URL from config (connector-specific logic)
        config = status.config
        url = self._build_service_url(spec, config)

        return MCPServerConfig(
            server_id=instance_id,
            name=status.name,
            url=url,
            enabled=status.connected and status.healthy,
        )

    # ── Suggestions ──────────────────────────────────────────────────

    def suggest_connectors(
        self, current_tools: list[str]
    ) -> list[dict[str, Any]]:
        """
        Suggest connectors based on what tools the user already has.
        E.g., if they have a database, suggest messaging for alerts.
        """
        suggestions = []
        connected_categories = {
            s.category.value for s in self._connections.values()
        }

        # Category affinity rules
        affinities = {
            "database": ["messaging", "storage", "monitoring"],
            "messaging": ["dev_tools", "api"],
            "dev_tools": ["database", "storage"],
            "storage": ["database", "monitoring"],
            "api": ["messaging", "storage"],
        }

        for cat in connected_categories:
            for suggested_cat in affinities.get(cat, []):
                if suggested_cat not in connected_categories:
                    # Find connectors in this category
                    for spec in self._catalog.values():
                        if spec.category.value == suggested_cat:
                            suggestions.append({
                                "connector_id": spec.connector_id,
                                "name": spec.name,
                                "category": spec.category.value,
                                "reason": f"Commonly paired with {cat} tools",
                                "icon": spec.icon,
                            })

        # Deduplicate
        seen = set()
        unique = []
        for s in suggestions:
            if s["connector_id"] not in seen:
                seen.add(s["connector_id"])
                unique.append(s)

        return unique[:5]  # Top 5 suggestions

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get registry-wide statistics."""
        total_conn = len(self._connections)
        healthy = sum(1 for s in self._connections.values() if s.healthy)
        total_calls = sum(s.total_calls for s in self._connections.values())
        total_errors = sum(s.error_count for s in self._connections.values())

        by_category: dict[str, int] = {}
        for s in self._connections.values():
            cat = s.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_connectors_available": len(self._catalog),
            "total_connections": total_conn,
            "healthy_connections": healthy,
            "unhealthy_connections": total_conn - healthy,
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": round(total_errors / max(total_calls, 1), 4),
            "connections_by_category": by_category,
        }

    # ── Internal Helpers ─────────────────────────────────────────────

    @staticmethod
    def _redact_secrets(
        config: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Redact secret fields before storing in status."""
        redacted = dict(config)
        for field_name, field_def in schema.items():
            if isinstance(field_def, dict) and field_def.get("secret") and field_name in redacted:
                val = str(redacted[field_name])
                redacted[field_name] = val[:4] + "****" if len(val) > 4 else "****"
        return redacted

    @staticmethod
    def _build_service_url(spec: ConnectorSpec, config: dict[str, Any]) -> str:
        """Build a service URL from connector spec and config."""
        if "base_url" in config:
            return config["base_url"]
        if "host" in config:
            port = config.get("port", spec.default_port or 80)
            return f"tcp://{config['host']}:{port}"
        if "db_path" in config:
            return f"file://{config['db_path']}"
        return f"shield://{spec.connector_id}"
