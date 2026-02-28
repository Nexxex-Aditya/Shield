"""
Shield Command — Living Connector Marketplace

A live marketplace for sharing, discovering, and health-tracking
forged connectors across Shield instances.

Features:
    - Publish connectors with usage stats and health scores
    - Search and discover connectors by capability
    - AI-suggested connectors based on pipeline descriptions
    - Health degradation alerts across all users
    - Featured/trending connectors based on community usage

Integration points:
    - ConnectorForge: publishes forged connectors
    - ForgeRegistry: pulls marketplace connectors into local registry
    - PipelineCompiler: suggests connectors during pipeline creation
"""

import json
import time
import uuid
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("shield.marketplace")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MarketplaceListing:
    """A connector listed in the marketplace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    connector_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    
    # Capabilities
    actions: list[str] = field(default_factory=list)  # action names
    action_count: int = 0
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    
    # Health & Usage
    health_score: float = 1.0      # 0=broken, 1=perfect
    total_installs: int = 0
    active_users: int = 0
    total_calls: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    
    # Status
    featured: bool = False
    verified: bool = False
    deprecated: bool = False
    
    published_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    spec_url: str = ""
    base_url: str = ""


@dataclass
class ConnectorSuggestion:
    """AI-suggested connector for a pipeline."""
    listing: MarketplaceListing
    reason: str = ""           # "Your pipeline mentions invoicing. This connector provides Stripe billing APIs."
    relevance: float = 0.0     # 0-1
    required_actions: list[str] = field(default_factory=list)  # Which specific actions would be useful


# ---------------------------------------------------------------------------
# Marketplace Engine
# ---------------------------------------------------------------------------

class ConnectorMarketplace:
    """
    Living marketplace for connector discovery and sharing.
    
    Usage:
        mp = ConnectorMarketplace()
        
        # Publish a connector
        await mp.publish(forged_connector, author="team-alpha")
        
        # Search
        results = await mp.search("payment processing stripe")
        
        # Get suggestions for a pipeline
        suggestions = await mp.suggest_for_pipeline("Monitor GitHub PRs and send Slack notifications")
        
        # Track health
        await mp.report_usage(connector_id="abc", success=True, latency_ms=450)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS marketplace_listings (
        id TEXT PRIMARY KEY,
        connector_id TEXT UNIQUE,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        version TEXT DEFAULT '1.0.0',
        author TEXT DEFAULT '',
        actions TEXT DEFAULT '[]',
        action_count INTEGER DEFAULT 0,
        categories TEXT DEFAULT '[]',
        tags TEXT DEFAULT '[]',
        health_score REAL DEFAULT 1.0,
        total_installs INTEGER DEFAULT 0,
        active_users INTEGER DEFAULT 0,
        total_calls INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 1.0,
        avg_latency_ms REAL DEFAULT 0.0,
        featured INTEGER DEFAULT 0,
        verified INTEGER DEFAULT 0,
        deprecated INTEGER DEFAULT 0,
        published_at REAL,
        updated_at REAL,
        spec_url TEXT DEFAULT '',
        base_url TEXT DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_mp_name ON marketplace_listings(name);
    CREATE INDEX IF NOT EXISTS idx_mp_health ON marketplace_listings(health_score);
    CREATE INDEX IF NOT EXISTS idx_mp_featured ON marketplace_listings(featured);
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Publishing ────────────────────────────────────────────────

    async def publish(self, connector, author: str = "anonymous") -> MarketplaceListing:
        """
        Publish a ForgedConnector to the marketplace.
        
        Args:
            connector: ForgedConnector instance
            author: publisher identifier
        """
        listing = MarketplaceListing(
            connector_id=connector.id,
            name=connector.name,
            description=connector.description,
            version=connector.version,
            author=author,
            actions=[a.name for a in connector.actions],
            action_count=len(connector.actions),
            tags=connector.tags,
            spec_url=connector.spec_url,
            base_url=connector.base_url,
        )

        # Auto-categorize
        listing.categories = self._categorize(connector)

        self._save_listing(listing)
        logger.info(f"Published '{listing.name}' to marketplace ({listing.action_count} actions)")
        return listing

    async def unpublish(self, connector_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM marketplace_listings WHERE connector_id = ?", (connector_id,))
        return True

    # ── Search & Discovery ────────────────────────────────────────

    async def search(
        self,
        query: str,
        min_health: float = 0.5,
        limit: int = 20,
    ) -> list[MarketplaceListing]:
        """Search marketplace by keyword."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM marketplace_listings 
                   WHERE (name LIKE ? OR description LIKE ? OR tags LIKE ? OR actions LIKE ?)
                   AND health_score >= ? AND deprecated = 0
                   ORDER BY total_installs DESC, health_score DESC 
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", min_health, limit),
            ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    async def get_featured(self, limit: int = 10) -> list[MarketplaceListing]:
        """Get featured connectors."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM marketplace_listings 
                   WHERE featured = 1 AND deprecated = 0
                   ORDER BY total_installs DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    async def get_trending(self, limit: int = 10) -> list[MarketplaceListing]:
        """Get trending connectors (most activity recently)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM marketplace_listings 
                   WHERE deprecated = 0 AND health_score >= 0.7
                   ORDER BY total_calls DESC, total_installs DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    async def suggest_for_pipeline(self, description: str) -> list[ConnectorSuggestion]:
        """
        Suggest connectors based on a pipeline description.
        
        Analyzes the description for service mentions and returns
        relevant marketplace listings.
        """
        service_keywords = {
            "github": ["github", "repository", "pr", "pull request", "commit", "branch"],
            "slack": ["slack", "notification", "message", "channel", "alert"],
            "email": ["email", "mail", "send email", "notification"],
            "stripe": ["payment", "billing", "invoice", "stripe", "charge"],
            "jira": ["jira", "ticket", "issue", "sprint", "backlog"],
            "database": ["database", "sql", "query", "table", "record"],
            "s3": ["s3", "storage", "upload", "file", "bucket"],
            "twilio": ["sms", "text message", "phone", "call", "twilio"],
            "hubspot": ["crm", "contact", "lead", "hubspot", "deal"],
            "notion": ["notion", "document", "wiki", "page"],
        }

        desc_lower = description.lower()
        suggested = []

        for service, keywords in service_keywords.items():
            matched_keywords = [kw for kw in keywords if kw in desc_lower]
            if matched_keywords:
                results = await self.search(service)
                for listing in results[:2]:
                    suggested.append(ConnectorSuggestion(
                        listing=listing,
                        reason=f"Your pipeline mentions '{matched_keywords[0]}'. {listing.name} provides relevant capabilities.",
                        relevance=len(matched_keywords) / len(keywords),
                    ))

        # Sort by relevance
        suggested.sort(key=lambda s: s.relevance, reverse=True)
        return suggested[:5]

    # ── Health Tracking ───────────────────────────────────────────

    async def report_usage(
        self,
        connector_id: str,
        success: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """Report a usage event to update health metrics."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT total_calls, success_rate, avg_latency_ms, health_score FROM marketplace_listings WHERE connector_id = ?",
                (connector_id,),
            ).fetchone()
            
            if not row:
                return
            
            total_calls = row[0] + 1
            old_success_rate = row[1]
            old_avg_latency = row[2]
            
            # Rolling average for success rate
            new_success_rate = (old_success_rate * row[0] + (1.0 if success else 0.0)) / total_calls
            
            # Rolling average for latency
            new_avg_latency = (old_avg_latency * row[0] + latency_ms) / total_calls
            
            # Health score: weighted combination
            new_health = new_success_rate * 0.7 + max(0, 1 - new_avg_latency / 5000) * 0.3
            
            conn.execute(
                """UPDATE marketplace_listings 
                   SET total_calls = ?, success_rate = ?, avg_latency_ms = ?, health_score = ?, updated_at = ?
                   WHERE connector_id = ?""",
                (total_calls, new_success_rate, new_avg_latency, new_health, time.time(), connector_id),
            )

    async def get_health_alerts(self, threshold: float = 0.7) -> list[MarketplaceListing]:
        """Get connectors with degraded health."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM marketplace_listings 
                   WHERE health_score < ? AND deprecated = 0 AND total_calls > 10
                   ORDER BY health_score ASC""",
                (threshold,),
            ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM marketplace_listings").fetchone()[0]
            featured = conn.execute("SELECT COUNT(*) FROM marketplace_listings WHERE featured = 1").fetchone()[0]
            total_installs = conn.execute("SELECT SUM(total_installs) FROM marketplace_listings").fetchone()[0] or 0
            avg_health = conn.execute("SELECT AVG(health_score) FROM marketplace_listings").fetchone()[0] or 0
        return {
            "total_listings": total,
            "featured_count": featured,
            "total_installs": total_installs,
            "avg_health_score": round(avg_health, 3),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _categorize(self, connector) -> list[str]:
        """Auto-categorize based on connector name and actions."""
        categories = []
        name_lower = connector.name.lower()
        
        category_keywords = {
            "communication": ["slack", "email", "twilio", "chat", "message"],
            "development": ["github", "gitlab", "bitbucket", "jira", "ci", "cd"],
            "data": ["database", "sql", "postgres", "mysql", "mongo", "redis"],
            "storage": ["s3", "gcs", "blob", "storage", "file"],
            "payment": ["stripe", "paypal", "billing", "payment"],
            "analytics": ["analytics", "metrics", "monitoring", "datadog"],
            "crm": ["hubspot", "salesforce", "crm", "contact"],
            "productivity": ["notion", "asana", "trello", "calendar"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in name_lower for kw in keywords):
                categories.append(category)
        
        return categories or ["general"]

    def _save_listing(self, listing: MarketplaceListing):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO marketplace_listings 
                   (id, connector_id, name, description, version, author, actions,
                    action_count, categories, tags, health_score, total_installs,
                    active_users, total_calls, success_rate, avg_latency_ms,
                    featured, verified, deprecated, published_at, updated_at,
                    spec_url, base_url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    listing.id, listing.connector_id, listing.name,
                    listing.description, listing.version, listing.author,
                    json.dumps(listing.actions), listing.action_count,
                    json.dumps(listing.categories), json.dumps(listing.tags),
                    listing.health_score, listing.total_installs,
                    listing.active_users, listing.total_calls,
                    listing.success_rate, listing.avg_latency_ms,
                    int(listing.featured), int(listing.verified),
                    int(listing.deprecated), listing.published_at,
                    listing.updated_at, listing.spec_url, listing.base_url,
                ),
            )

    def _row_to_listing(self, row: tuple) -> MarketplaceListing:
        return MarketplaceListing(
            id=row[0], connector_id=row[1], name=row[2], description=row[3],
            version=row[4], author=row[5], actions=json.loads(row[6] or "[]"),
            action_count=row[7], categories=json.loads(row[8] or "[]"),
            tags=json.loads(row[9] or "[]"), health_score=row[10],
            total_installs=row[11], active_users=row[12], total_calls=row[13],
            success_rate=row[14], avg_latency_ms=row[15],
            featured=bool(row[16]), verified=bool(row[17]),
            deprecated=bool(row[18]), published_at=row[19], updated_at=row[20],
            spec_url=row[21], base_url=row[22],
        )
