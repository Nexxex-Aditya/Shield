"""
Shield Command — Universal Connector Forge

Generates production-ready connectors from OpenAPI/Swagger specifications.
Instead of maintaining 1000 hand-written integrations, Shield generates
them automatically from any API's spec.

Workflow:
    1. User provides an OpenAPI spec URL or JSON/YAML content
    2. ConnectorForge parses the spec (paths, methods, auth, schemas)
    3. Generates a ForgedConnector with all actions auto-mapped
    4. Connector is immediately usable in pipelines and agent tools
    5. Stored in ForgeRegistry for persistence and marketplace publishing

Integration points:
    - BaseConnector: ForgedConnector extends the existing connector base
    - ConnectorExecutor: auto-registers forged connectors
    - MCPGateway: forged connector actions become available as tools
    - Marketplace: publish/discover forged connectors
"""

import json
import re
import uuid
import time
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger("shield.connector_forge")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ForgedAction:
    """A single API action generated from an OpenAPI operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""                # operation_id or generated name
    method: str = "GET"           # HTTP method
    path: str = ""                # URL path (with placeholders)
    description: str = ""
    parameters: list[dict] = field(default_factory=list)  # query/path params
    request_body: Optional[dict] = None                    # body schema
    response_schema: Optional[dict] = None
    required_params: list[str] = field(default_factory=list)
    auth_required: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class AuthConfig:
    """Authentication configuration extracted from OpenAPI spec."""
    type: str = "none"              # none | api_key | bearer | oauth2 | basic
    key_name: str = ""              # Header/query param name for API key
    key_location: str = "header"    # header | query | cookie
    token_url: str = ""             # For OAuth2
    scopes: list[str] = field(default_factory=list)


@dataclass
class ForgedConnector:
    """A complete connector generated from an OpenAPI spec."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    base_url: str = ""
    auth: AuthConfig = field(default_factory=AuthConfig)
    actions: list[ForgedAction] = field(default_factory=list)
    spec_url: str = ""
    created_at: float = field(default_factory=time.time)
    health_score: float = 1.0
    usage_count: int = 0
    tags: list[str] = field(default_factory=list)
    raw_spec_hash: str = ""

    def get_action(self, name: str) -> Optional[ForgedAction]:
        for a in self.actions:
            if a.name == name:
                return a
        return None

    def list_actions(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "method": a.method,
                "path": a.path,
                "description": a.description,
                "params": [p.get("name") for p in a.parameters],
            }
            for a in self.actions
        ]


# ---------------------------------------------------------------------------
# OpenAPI Parser
# ---------------------------------------------------------------------------

class OpenAPIParser:
    """
    Parses OpenAPI 3.x and Swagger 2.x specifications into structured data.
    Handles JSON and YAML formats.
    """

    def parse(self, spec_data: dict) -> dict:
        """Parse a raw OpenAPI spec dict into normalized structure."""
        version = spec_data.get("openapi", spec_data.get("swagger", ""))
        
        if version.startswith("3"):
            return self._parse_v3(spec_data)
        elif version.startswith("2"):
            return self._parse_v2(spec_data)
        else:
            raise ValueError(f"Unsupported OpenAPI version: {version}")

    def _parse_v3(self, spec: dict) -> dict:
        """Parse OpenAPI 3.x spec."""
        info = spec.get("info", {})
        servers = spec.get("servers", [])
        paths = spec.get("paths", {})
        security_schemes = (
            spec.get("components", {}).get("securitySchemes", {})
        )

        return {
            "title": info.get("title", "Unknown API"),
            "description": info.get("description", ""),
            "version": info.get("version", "1.0.0"),
            "base_url": servers[0].get("url", "") if servers else "",
            "paths": self._parse_paths(paths, spec),
            "auth": self._parse_security_schemes(security_schemes),
            "tags": [t.get("name", "") for t in spec.get("tags", [])],
        }

    def _parse_v2(self, spec: dict) -> dict:
        """Parse Swagger 2.x spec."""
        info = spec.get("info", {})
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        schemes = spec.get("schemes", ["https"])
        paths = spec.get("paths", {})
        security_defs = spec.get("securityDefinitions", {})

        base_url = f"{schemes[0]}://{host}{base_path}" if host else ""

        return {
            "title": info.get("title", "Unknown API"),
            "description": info.get("description", ""),
            "version": info.get("version", "1.0.0"),
            "base_url": base_url,
            "paths": self._parse_paths(paths, spec),
            "auth": self._parse_security_defs_v2(security_defs),
            "tags": [t.get("name", "") for t in spec.get("tags", [])],
        }

    def _parse_paths(self, paths: dict, spec: dict) -> list[dict]:
        """Parse all paths into operations."""
        operations = []
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                    continue
                if not isinstance(operation, dict):
                    continue

                op_id = operation.get("operationId") or self._generate_op_id(method, path)
                
                # Extract parameters
                params = []
                for p in operation.get("parameters", []) + methods.get("parameters", []):
                    if isinstance(p, dict):
                        # Resolve $ref if needed
                        p = self._resolve_ref(p, spec)
                        params.append({
                            "name": p.get("name", ""),
                            "in": p.get("in", "query"),
                            "required": p.get("required", False),
                            "type": self._extract_type(p.get("schema", p)),
                            "description": p.get("description", ""),
                        })

                # Extract request body (v3)
                body = None
                request_body = operation.get("requestBody", {})
                if request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    body = json_content.get("schema")
                    if body:
                        body = self._resolve_ref(body, spec)

                operations.append({
                    "operation_id": op_id,
                    "method": method.upper(),
                    "path": path,
                    "summary": operation.get("summary", ""),
                    "description": operation.get("description", ""),
                    "parameters": params,
                    "request_body": body,
                    "tags": operation.get("tags", []),
                    "required_params": [p["name"] for p in params if p.get("required")],
                })

        return operations

    def _parse_security_schemes(self, schemes: dict) -> AuthConfig:
        """Parse OpenAPI 3.x security schemes."""
        for name, scheme in schemes.items():
            scheme_type = scheme.get("type", "")
            if scheme_type == "apiKey":
                return AuthConfig(
                    type="api_key",
                    key_name=scheme.get("name", "X-API-Key"),
                    key_location=scheme.get("in", "header"),
                )
            elif scheme_type == "http":
                sub = scheme.get("scheme", "").lower()
                if sub == "bearer":
                    return AuthConfig(type="bearer")
                elif sub == "basic":
                    return AuthConfig(type="basic")
            elif scheme_type == "oauth2":
                flows = scheme.get("flows", {})
                cc_flow = flows.get("clientCredentials", flows.get("authorizationCode", {}))
                return AuthConfig(
                    type="oauth2",
                    token_url=cc_flow.get("tokenUrl", ""),
                    scopes=list(cc_flow.get("scopes", {}).keys()),
                )
        return AuthConfig(type="none")

    def _parse_security_defs_v2(self, defs: dict) -> AuthConfig:
        """Parse Swagger 2.x security definitions."""
        for name, defn in defs.items():
            dtype = defn.get("type", "")
            if dtype == "apiKey":
                return AuthConfig(
                    type="api_key",
                    key_name=defn.get("name", "api_key"),
                    key_location=defn.get("in", "header"),
                )
            elif dtype == "oauth2":
                return AuthConfig(
                    type="oauth2",
                    token_url=defn.get("tokenUrl", ""),
                    scopes=list(defn.get("scopes", {}).keys()),
                )
        return AuthConfig(type="none")

    def _resolve_ref(self, obj: dict, spec: dict) -> dict:
        """Resolve a $ref pointer."""
        ref = obj.get("$ref")
        if not ref:
            return obj
        parts = ref.lstrip("#/").split("/")
        resolved = spec
        for part in parts:
            resolved = resolved.get(part, {})
        return resolved if isinstance(resolved, dict) else obj

    def _generate_op_id(self, method: str, path: str) -> str:
        """Generate a clean operation ID from method + path."""
        clean = re.sub(r"[{}]", "", path)
        clean = re.sub(r"[^a-zA-Z0-9/]", "", clean)
        parts = [p for p in clean.split("/") if p]
        return f"{method.lower()}_{'_'.join(parts)}" if parts else f"{method.lower()}_root"

    def _extract_type(self, schema: dict) -> str:
        """Extract a simple type string from a schema."""
        if not isinstance(schema, dict):
            return "string"
        return schema.get("type", schema.get("format", "string"))


# ---------------------------------------------------------------------------
# Connector Forge — The Main Engine
# ---------------------------------------------------------------------------

class ConnectorForge:
    """
    Generate production-ready connectors from OpenAPI specifications.
    
    Usage:
        forge = ConnectorForge()
        connector = await forge.forge_from_url("https://api.example.com/openapi.json")
        # or
        connector = await forge.forge_from_spec(spec_dict)
        
        # The connector is immediately usable
        actions = connector.list_actions()
    """

    def __init__(self):
        self.parser = OpenAPIParser()

    async def forge_from_url(self, spec_url: str) -> ForgedConnector:
        """Fetch an OpenAPI spec from a URL and generate a connector."""
        import aiohttp

        logger.info(f"Forging connector from URL: {spec_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(spec_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        raise ValueError(f"Failed to fetch spec: HTTP {resp.status}")
                    content_type = resp.headers.get("Content-Type", "")
                    text = await resp.text()
        except ImportError:
            # Fallback to urllib if aiohttp not available
            import urllib.request
            with urllib.request.urlopen(spec_url, timeout=30) as resp:
                text = resp.read().decode("utf-8")
                content_type = resp.headers.get("Content-Type", "")

        # Parse JSON or YAML
        if "yaml" in content_type or spec_url.endswith((".yaml", ".yml")):
            try:
                import yaml
                spec_data = yaml.safe_load(text)
            except ImportError:
                raise ValueError("YAML spec detected but PyYAML not installed")
        else:
            spec_data = json.loads(text)

        connector = await self.forge_from_spec(spec_data)
        connector.spec_url = spec_url
        return connector

    async def forge_from_spec(self, spec_data: dict) -> ForgedConnector:
        """Generate a connector from a parsed OpenAPI spec dictionary."""
        parsed = self.parser.parse(spec_data)
        
        connector = ForgedConnector(
            name=parsed["title"],
            description=parsed["description"],
            version=parsed["version"],
            base_url=parsed["base_url"],
            auth=parsed["auth"],
            tags=parsed["tags"],
            raw_spec_hash=str(hash(json.dumps(spec_data, sort_keys=True, default=str)))[:16],
        )

        # Generate actions from parsed operations
        for op in parsed["paths"]:
            action = ForgedAction(
                name=op["operation_id"],
                method=op["method"],
                path=op["path"],
                description=op.get("summary") or op.get("description", ""),
                parameters=op["parameters"],
                request_body=op.get("request_body"),
                required_params=op["required_params"],
                tags=op.get("tags", []),
            )
            connector.actions.append(action)

        logger.info(
            f"Forged connector '{connector.name}' with {len(connector.actions)} actions "
            f"(auth: {connector.auth.type})"
        )
        return connector

    async def forge_from_json(self, json_str: str) -> ForgedConnector:
        """Generate a connector from a JSON string."""
        spec_data = json.loads(json_str)
        return await self.forge_from_spec(spec_data)


# ---------------------------------------------------------------------------
# Forge Registry — Persistence layer
# ---------------------------------------------------------------------------

class ForgeRegistry:
    """
    SQLite-backed registry of forged connectors.
    Supports save, load, search, and health tracking.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS forged_connectors (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        version TEXT DEFAULT '1.0.0',
        base_url TEXT DEFAULT '',
        auth_config TEXT DEFAULT '{}',
        actions TEXT DEFAULT '[]',
        spec_url TEXT DEFAULT '',
        created_at REAL NOT NULL,
        health_score REAL DEFAULT 1.0,
        usage_count INTEGER DEFAULT 0,
        tags TEXT DEFAULT '[]',
        raw_spec_hash TEXT DEFAULT '',
        published INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_forge_name ON forged_connectors(name);
    CREATE INDEX IF NOT EXISTS idx_forge_published ON forged_connectors(published);
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def save(self, connector: ForgedConnector) -> str:
        """Save a forged connector to the registry."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO forged_connectors 
                   (id, name, description, version, base_url, auth_config, 
                    actions, spec_url, created_at, health_score, usage_count, tags, raw_spec_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    connector.id, connector.name, connector.description,
                    connector.version, connector.base_url,
                    json.dumps(asdict(connector.auth)),
                    json.dumps([asdict(a) for a in connector.actions]),
                    connector.spec_url, connector.created_at,
                    connector.health_score, connector.usage_count,
                    json.dumps(connector.tags), connector.raw_spec_hash,
                ),
            )
        logger.info(f"Saved connector '{connector.name}' ({connector.id})")
        return connector.id

    async def load(self, connector_id: str) -> Optional[ForgedConnector]:
        """Load a connector from the registry."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM forged_connectors WHERE id = ?", (connector_id,)
            ).fetchone()
        return self._row_to_connector(row) if row else None

    async def find_by_name(self, name: str) -> Optional[ForgedConnector]:
        """Find a connector by name."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM forged_connectors WHERE name LIKE ? LIMIT 1",
                (f"%{name}%",),
            ).fetchone()
        return self._row_to_connector(row) if row else None

    async def list_all(self, published_only: bool = False) -> list[ForgedConnector]:
        """List all connectors in the registry."""
        with self._conn() as conn:
            condition = "WHERE published = 1" if published_only else ""
            rows = conn.execute(
                f"SELECT * FROM forged_connectors {condition} ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_connector(row) for row in rows]

    async def search(self, query: str) -> list[ForgedConnector]:
        """Search connectors by name, description, or tags."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM forged_connectors 
                   WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?
                   ORDER BY usage_count DESC LIMIT 20""",
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            ).fetchall()
        return [self._row_to_connector(row) for row in rows]

    async def record_usage(self, connector_id: str, success: bool = True):
        """Track connector usage and update health score."""
        with self._conn() as conn:
            if success:
                conn.execute(
                    """UPDATE forged_connectors 
                       SET usage_count = usage_count + 1,
                           health_score = MIN(health_score + 0.01, 1.0)
                       WHERE id = ?""",
                    (connector_id,),
                )
            else:
                conn.execute(
                    """UPDATE forged_connectors 
                       SET usage_count = usage_count + 1,
                           health_score = MAX(health_score - 0.05, 0.0)
                       WHERE id = ?""",
                    (connector_id,),
                )

    async def publish(self, connector_id: str) -> bool:
        """Mark a connector as published (available in marketplace)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE forged_connectors SET published = 1 WHERE id = ?",
                (connector_id,),
            )
        return True

    async def delete(self, connector_id: str) -> bool:
        """Remove a connector from the registry."""
        with self._conn() as conn:
            conn.execute("DELETE FROM forged_connectors WHERE id = ?", (connector_id,))
        return True

    async def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM forged_connectors").fetchone()[0]

    def _row_to_connector(self, row: tuple) -> ForgedConnector:
        auth_data = json.loads(row[5] or "{}")
        actions_data = json.loads(row[6] or "[]")
        
        actions = [
            ForgedAction(
                id=a.get("id", ""),
                name=a.get("name", ""),
                method=a.get("method", "GET"),
                path=a.get("path", ""),
                description=a.get("description", ""),
                parameters=a.get("parameters", []),
                request_body=a.get("request_body"),
                required_params=a.get("required_params", []),
                tags=a.get("tags", []),
            )
            for a in actions_data
        ]

        return ForgedConnector(
            id=row[0], name=row[1], description=row[2], version=row[3],
            base_url=row[4],
            auth=AuthConfig(**auth_data) if auth_data else AuthConfig(),
            actions=actions, spec_url=row[7], created_at=row[8],
            health_score=row[9], usage_count=row[10],
            tags=json.loads(row[11] or "[]"), raw_spec_hash=row[12],
        )


# ---------------------------------------------------------------------------
# HTTP Executor for Forged Connectors
# ---------------------------------------------------------------------------

class ForgedConnectorExecutor:
    """
    Executes actions on forged connectors by making real HTTP calls.
    Handles authentication, parameter injection, and error handling.
    """

    def __init__(self, registry: ForgeRegistry):
        self.registry = registry

    async def execute_action(
        self,
        connector_id: str,
        action_name: str,
        params: dict,
        credentials: Optional[dict] = None,
    ) -> dict:
        """Execute a forged connector action."""
        connector = await self.registry.load(connector_id)
        if not connector:
            return {"error": f"Connector {connector_id} not found", "success": False}

        action = connector.get_action(action_name)
        if not action:
            return {"error": f"Action {action_name} not found in {connector.name}", "success": False}

        # Validate required parameters
        missing = [p for p in action.required_params if p not in params]
        if missing:
            return {"error": f"Missing required parameters: {missing}", "success": False}

        # Build the URL
        url = connector.base_url.rstrip("/") + action.path
        for param in action.parameters:
            if param.get("in") == "path" and param["name"] in params:
                url = url.replace(f"{{{param['name']}}}", str(params[param["name"]]))

        # Separate query params from body params
        query_params = {}
        body_data = {}
        for param in action.parameters:
            name = param["name"]
            if name in params:
                if param.get("in") == "query":
                    query_params[name] = params[name]
                elif param.get("in") not in ("path", "header"):
                    body_data[name] = params[name]

        # Add any extra params to body for POST/PUT/PATCH
        if action.method in ("POST", "PUT", "PATCH"):
            for k, v in params.items():
                if k not in query_params and k not in [p["name"] for p in action.parameters if p.get("in") == "path"]:
                    body_data[k] = v

        # Build headers with auth
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if credentials:
            if connector.auth.type == "api_key":
                if connector.auth.key_location == "header":
                    headers[connector.auth.key_name] = credentials.get("api_key", "")
                else:
                    query_params[connector.auth.key_name] = credentials.get("api_key", "")
            elif connector.auth.type == "bearer":
                headers["Authorization"] = f"Bearer {credentials.get('token', '')}"
            elif connector.auth.type == "basic":
                import base64
                creds = base64.b64encode(
                    f"{credentials.get('username', '')}:{credentials.get('password', '')}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {creds}"

        # Execute the HTTP request
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                kwargs = {"headers": headers, "params": query_params}
                if body_data and action.method in ("POST", "PUT", "PATCH"):
                    kwargs["json"] = body_data

                async with session.request(
                    action.method, url, timeout=aiohttp.ClientTimeout(total=30), **kwargs
                ) as resp:
                    status = resp.status
                    try:
                        result = await resp.json()
                    except Exception:
                        result = await resp.text()

            success = 200 <= status < 300
            await self.registry.record_usage(connector_id, success=success)
            
            return {
                "success": success,
                "status_code": status,
                "data": result,
                "connector": connector.name,
                "action": action_name,
            }

        except ImportError:
            return {
                "error": "aiohttp not installed — cannot execute HTTP requests",
                "success": False,
                "connector": connector.name,
                "action": action_name,
            }
        except Exception as e:
            await self.registry.record_usage(connector_id, success=False)
            return {
                "error": str(e),
                "success": False,
                "connector": connector.name,
                "action": action_name,
            }
