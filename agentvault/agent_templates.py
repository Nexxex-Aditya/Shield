"""
Shield Command — Agent Template Genome

Describe an agent in English → get a complete, shareable template
containing pipeline, connectors, policies, and model configuration.

Templates are evolvable: Agent Genetics can optimize them.
Templates are shareable: install from marketplace like apps.

Integration points:
    - ModelRegistry: generates templates via LLM
    - ConnectorForge: auto-forges needed connectors
    - NLPolicyCompiler: compiles policies from template description
    - EvolutionEngine: evolves installed templates
    - ConnectorMarketplace: publishes templates
"""

import json
import time
import uuid
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("shield.agent_templates")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TemplateConfig:
    """Complete agent configuration template."""
    # Identity
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)

    # Model configuration
    preferred_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""

    # Pipeline steps
    pipeline_steps: list[dict] = field(default_factory=list)

    # Required connectors
    connectors: list[dict] = field(default_factory=list)

    # Security policies (natural language)
    policies: list[str] = field(default_factory=list)

    # Default parameters
    default_params: dict = field(default_factory=dict)


@dataclass
class AgentTemplate:
    """A shareable, installable agent template."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    config: TemplateConfig = field(default_factory=TemplateConfig)
    
    # Marketplace metadata
    installs: int = 0
    rating: float = 0.0
    reviews: int = 0
    
    # Status
    published: bool = False
    verified: bool = False
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Template Generator
# ---------------------------------------------------------------------------

class TemplateGenerator:
    """
    Generates complete agent templates from English descriptions.
    
    Usage:
        gen = TemplateGenerator(model_registry=registry)
        template = await gen.generate(
            "An agent that monitors GitHub PRs, runs code review,
             and posts results to Slack. Should not approve PRs
             with security vulnerabilities."
        )
    """

    def __init__(self, model_registry=None):
        self.model_registry = model_registry

    async def generate(self, description: str, author: str = "user") -> AgentTemplate:
        """Generate a template from an English description."""
        config = TemplateConfig(description=description, author=author)

        # Try LLM generation
        if self.model_registry:
            try:
                config = await self._generate_with_llm(description, author)
            except Exception as e:
                logger.warning(f"LLM template generation failed, using pattern-based: {e}")
                config = self._generate_with_patterns(description, author)
        else:
            config = self._generate_with_patterns(description, author)

        return AgentTemplate(config=config)

    async def _generate_with_llm(self, description: str, author: str) -> TemplateConfig:
        """Use LLM to generate a complete template."""
        adapter = self.model_registry.get_adapter("reasoning")
        if not adapter:
            raise ValueError("No reasoning adapter")

        response = await adapter.generate(
            system_prompt=(
                "You are an AI agent architect. Generate a complete agent template from the description.\n"
                "Output ONLY valid JSON:\n"
                '{\n'
                '  "name": "short agent name",\n'
                '  "description": "what this agent does",\n'
                '  "tags": ["category1", "category2"],\n'
                '  "preferred_model": "gpt-4 or claude-3-sonnet or gemini-pro",\n'
                '  "temperature": 0.7,\n'
                '  "system_prompt": "You are an agent that...",\n'
                '  "pipeline_steps": [\n'
                '    {"name": "step1", "type": "tool_call", "tool_name": "...", "description": "..."},\n'
                '    {"name": "step2", "type": "llm_call", "description": "..."}\n'
                '  ],\n'
                '  "connectors": [\n'
                '    {"service": "github", "required_actions": ["list_prs", "post_comment"]}\n'
                '  ],\n'
                '  "policies": [\n'
                '    "Never approve PRs with known CVEs",\n'
                '    "Rate limit to 50 reviews per hour"\n'
                '  ]\n'
                "}"
            ),
            user_prompt=f"Description: {description}",
            temperature=0.3,
            max_tokens=800,
        )

        content = response.get("content", "")
        parsed = self._parse_json(content)

        return TemplateConfig(
            name=parsed.get("name", description[:40]),
            description=parsed.get("description", description),
            author=author,
            tags=parsed.get("tags", []),
            preferred_model=parsed.get("preferred_model", ""),
            temperature=parsed.get("temperature", 0.7),
            system_prompt=parsed.get("system_prompt", ""),
            pipeline_steps=parsed.get("pipeline_steps", []),
            connectors=parsed.get("connectors", []),
            policies=parsed.get("policies", []),
        )

    def _generate_with_patterns(self, description: str, author: str) -> TemplateConfig:
        """Pattern-based template generation."""
        desc_lower = description.lower()
        config = TemplateConfig(
            name=description[:40],
            description=description,
            author=author,
        )

        # Detect services
        service_map = {
            "github": {"service": "github", "actions": ["list_prs", "create_issue", "post_comment"]},
            "slack": {"service": "slack", "actions": ["send_message", "post_notification"]},
            "email": {"service": "email", "actions": ["send_email"]},
            "jira": {"service": "jira", "actions": ["create_ticket", "update_ticket"]},
            "database": {"service": "database", "actions": ["query", "insert"]},
        }

        for service, details in service_map.items():
            if service in desc_lower:
                config.connectors.append(details)
                config.tags.append(service)

        # Detect pipeline type
        if any(w in desc_lower for w in ["monitor", "watch", "track", "alert"]):
            config.pipeline_steps.append({"name": "monitor", "type": "trigger", "description": "Watch for events"})
            config.tags.append("monitoring")

        if any(w in desc_lower for w in ["analyze", "review", "check", "evaluate"]):
            config.pipeline_steps.append({"name": "analyze", "type": "llm_call", "description": "Analyze data"})
            config.tags.append("analysis")

        if any(w in desc_lower for w in ["notify", "send", "report", "post"]):
            config.pipeline_steps.append({"name": "notify", "type": "tool_call", "description": "Send notification"})
            config.tags.append("notification")

        # Detect security policies
        if any(w in desc_lower for w in ["security", "vulnerability", "safe", "restrict"]):
            config.policies.append("Block actions that introduce security vulnerabilities")

        if any(w in desc_lower for w in ["limit", "rate", "throttle"]):
            config.policies.append("Rate limit API calls to prevent abuse")

        config.system_prompt = f"You are an AI agent. Your task: {description}"
        return config

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}


# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------

class TemplateRegistry:
    """
    Persistence for agent templates. Save, load, search, share.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_templates (
        id TEXT PRIMARY KEY,
        config TEXT NOT NULL,
        installs INTEGER DEFAULT 0,
        rating REAL DEFAULT 0,
        reviews INTEGER DEFAULT 0,
        published INTEGER DEFAULT 0,
        verified INTEGER DEFAULT 0,
        created_at REAL,
        updated_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_at_published ON agent_templates(published);
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def save(self, template: AgentTemplate) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO agent_templates 
                   (id, config, installs, rating, reviews, published, verified, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (template.id, json.dumps(asdict(template.config), default=str),
                 template.installs, template.rating, template.reviews,
                 int(template.published), int(template.verified),
                 template.created_at, time.time()),
            )
        return template.id

    async def load(self, template_id: str) -> Optional[AgentTemplate]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agent_templates WHERE id = ?", (template_id,)
            ).fetchone()
        if not row:
            return None
        config_data = json.loads(row[1])
        return AgentTemplate(
            id=row[0],
            config=TemplateConfig(**{k: v for k, v in config_data.items() if k in TemplateConfig.__dataclass_fields__}),
            installs=row[2], rating=row[3], reviews=row[4],
            published=bool(row[5]), verified=bool(row[6]),
            created_at=row[7], updated_at=row[8],
        )

    async def search(self, query: str, limit: int = 20) -> list[AgentTemplate]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM agent_templates WHERE config LIKE ? ORDER BY installs DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        results = []
        for row in rows:
            t = await self.load(row[0])
            if t:
                results.append(t)
        return results

    async def list_all(self, published_only: bool = False) -> list[AgentTemplate]:
        with self._conn() as conn:
            condition = "WHERE published = 1" if published_only else ""
            rows = conn.execute(f"SELECT id FROM agent_templates {condition} ORDER BY created_at DESC").fetchall()
        results = []
        for row in rows:
            t = await self.load(row[0])
            if t:
                results.append(t)
        return results

    async def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM agent_templates").fetchone()[0]
            published = conn.execute("SELECT COUNT(*) FROM agent_templates WHERE published = 1").fetchone()[0]
            total_installs = conn.execute("SELECT SUM(installs) FROM agent_templates").fetchone()[0] or 0
        return {"total": total, "published": published, "total_installs": total_installs}
