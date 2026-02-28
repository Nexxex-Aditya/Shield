"""
AgentVault — Skills Engine
Schema-driven AI runbooks that teach agents how to execute tasks.

Customers create skills via the dashboard or YAML files. Each skill is a
structured set of steps with security constraints. The engine validates,
stores, and connects skills to the MCP gateway so AI agents can discover
and follow them at runtime.

Security model:
    - Content sanitization (strip injection patterns)
    - Tool allowlist/blocklist per skill
    - Size limits (max steps, max instruction length)
    - Execution scoping (skills can't escalate permissions)
    - Full audit logging of all operations
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("agentvault.skills")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 50
MAX_INSTRUCTION_LEN = 5000
MAX_SKILL_NAME_LEN = 128
MAX_DESCRIPTION_LEN = 1000
MAX_TAGS = 20

# Injection patterns to strip from skill content
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+(?:a|an|in)\s+",
    r"system\s*:\s*",
    r"<\s*(?:script|img|iframe|object|embed|form)",
    r"javascript\s*:",
    r"data\s*:\s*text/html",
    r"on(?:load|error|click|mouseover)\s*=",
    r"\{\{\s*",  # template injection
    r"\$\{",      # template literal injection
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"os\.system",
    r"subprocess",
    r"rm\s+-rf",
    r"DROP\s+TABLE",
    r";\s*--",     # SQL injection
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


# ---------------------------------------------------------------------------
# Schema Models
# ---------------------------------------------------------------------------

class SkillPermissions(BaseModel):
    """Security constraints for a skill."""
    max_execution_time: int = Field(default=300, ge=10, le=3600)
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=lambda: ["delete_*", "admin_*", "drop_*"])
    require_approval: bool = False
    max_retries: int = Field(default=3, ge=0, le=10)
    sandbox_mode: bool = True


class SkillStep(BaseModel):
    """A single step in a skill."""
    title: str = Field(..., min_length=1, max_length=256)
    instruction: str = Field(..., min_length=1, max_length=MAX_INSTRUCTION_LEN)
    tools_needed: list[str] = Field(default_factory=list)
    expected_output: str = Field(default="", max_length=1000)
    on_failure: str = Field(default="stop", pattern=r"^(stop|skip|retry)$")
    order: int = Field(default=0, ge=0)

    @field_validator("instruction")
    @classmethod
    def sanitize_instruction(cls, v: str) -> str:
        return _sanitize_content(v)


class SkillSchema(BaseModel):
    """Full skill definition — what customers fill in."""
    name: str = Field(..., min_length=1, max_length=MAX_SKILL_NAME_LEN,
                      pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$")
    description: str = Field(..., min_length=1, max_length=MAX_DESCRIPTION_LEN)
    version: str = Field(default="1.0", max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)
    author: str = Field(default="system", max_length=128)
    steps: list[SkillStep] = Field(..., min_length=1, max_length=MAX_STEPS)
    permissions: SkillPermissions = Field(default_factory=SkillPermissions)

    # Metadata (auto-populated)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str = Field(default="")
    enabled: bool = True

    @field_validator("description")
    @classmethod
    def sanitize_description(cls, v: str) -> str:
        return _sanitize_content(v)

    def compute_hash(self) -> str:
        """Content-addressable hash for integrity checking."""
        payload = json.dumps({
            "name": self.name,
            "steps": [s.model_dump() for s in self.steps],
            "permissions": self.permissions.model_dump(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Security Helpers
# ---------------------------------------------------------------------------

def _sanitize_content(text: str) -> str:
    """Strip known injection patterns from skill content."""
    cleaned = text
    for pattern in _COMPILED_PATTERNS:
        cleaned = pattern.sub("[FILTERED]", cleaned)
    return cleaned


def _validate_tool_permissions(skill: SkillSchema) -> list[str]:
    """
    Check that step tools don't violate the skill's permission constraints.
    Returns list of violation messages (empty = valid).
    """
    violations = []
    allowed = set(skill.permissions.allowed_tools) if skill.permissions.allowed_tools else None
    blocked_patterns = skill.permissions.blocked_tools

    for i, step in enumerate(skill.steps):
        for tool in step.tools_needed:
            # Check blocklist
            for bp in blocked_patterns:
                if bp.endswith("*"):
                    prefix = bp[:-1]
                    if tool.startswith(prefix):
                        violations.append(
                            f"Step {i+1} ({step.title}): tool '{tool}' matches blocked pattern '{bp}'"
                        )
                elif tool == bp:
                    violations.append(
                        f"Step {i+1} ({step.title}): tool '{tool}' is blocked"
                    )
            # Check allowlist (if specified)
            if allowed and tool not in allowed:
                violations.append(
                    f"Step {i+1} ({step.title}): tool '{tool}' not in allowed_tools"
                )
    return violations


# ---------------------------------------------------------------------------
# Skills Engine
# ---------------------------------------------------------------------------

class SkillsEngine:
    """
    Manages skill lifecycle — create, validate, store, load, connect.

    Skills are stored as YAML files in the skills directory and loaded
    into memory on startup. The engine exposes methods for CRUD operations
    from the dashboard and API, and provides structured output for AI agents.
    """

    def __init__(self, skills_dir: str = "skills") -> None:
        self._skills_dir = Path(skills_dir)
        self._registry: dict[str, SkillSchema] = {}
        self._audit_log: list[dict] = []
        self._execution_count: dict[str, int] = {}

    # ── Lifecycle ────────────────────────────────────────────────────

    def load_directory(self) -> int:
        """Load all YAML skill files from the skills directory."""
        if not self._skills_dir.exists():
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created skills directory: %s", self._skills_dir)
            return 0

        loaded = 0
        for fp in sorted(self._skills_dir.glob("*.yaml")):
            try:
                self.import_skill(fp.read_text(encoding="utf-8"), source=str(fp))
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load skill %s: %s", fp.name, e)
        logger.info("Loaded %d skills from %s", loaded, self._skills_dir)
        return loaded

    # ── CRUD ─────────────────────────────────────────────────────────

    def create_skill(self, data: dict) -> SkillSchema:
        """Create and register a new skill from raw data."""
        schema = SkillSchema(**data)

        if schema.name in self._registry:
            raise ValueError(f"Skill '{schema.name}' already exists. Use update_skill().")

        # Security validation
        violations = _validate_tool_permissions(schema)
        if violations:
            raise ValueError(f"Permission violations: {'; '.join(violations)}")

        # Compute integrity hash
        schema.content_hash = schema.compute_hash()
        schema.created_at = datetime.utcnow().isoformat()
        schema.updated_at = schema.created_at

        # Register
        self._registry[schema.name] = schema
        self._save_to_disk(schema)
        self._log_operation("create", schema.name)
        logger.info("Created skill: %s (v%s, %d steps)", schema.name, schema.version, len(schema.steps))
        return schema

    def update_skill(self, name: str, data: dict) -> SkillSchema:
        """Update an existing skill."""
        if name not in self._registry:
            raise KeyError(f"Skill '{name}' not found")

        existing = self._registry[name]
        # Merge: keep original created_at, author
        data.setdefault("name", name)
        data.setdefault("author", existing.author)
        data.setdefault("created_at", existing.created_at)
        data["updated_at"] = datetime.utcnow().isoformat()

        schema = SkillSchema(**data)
        violations = _validate_tool_permissions(schema)
        if violations:
            raise ValueError(f"Permission violations: {'; '.join(violations)}")

        schema.content_hash = schema.compute_hash()
        self._registry[name] = schema
        self._save_to_disk(schema)
        self._log_operation("update", name)
        logger.info("Updated skill: %s (v%s)", name, schema.version)
        return schema

    def delete_skill(self, name: str) -> bool:
        """Remove a skill from registry and disk."""
        if name not in self._registry:
            raise KeyError(f"Skill '{name}' not found")

        del self._registry[name]
        fp = self._skills_dir / f"{name}.yaml"
        if fp.exists():
            fp.unlink()
        self._log_operation("delete", name)
        logger.info("Deleted skill: %s", name)
        return True

    def get_skill(self, name: str) -> Optional[SkillSchema]:
        """Retrieve a skill by name."""
        return self._registry.get(name)

    def list_skills(self) -> list[dict]:
        """List all registered skills (summary view)."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "tags": s.tags,
                "author": s.author,
                "steps_count": len(s.steps),
                "enabled": s.enabled,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "content_hash": s.content_hash,
                "execution_count": self._execution_count.get(s.name, 0),
            }
            for s in self._registry.values()
        ]

    # ── Execution ────────────────────────────────────────────────────

    def execute_skill(self, name: str, context: dict = None) -> dict:
        """
        Prepare a skill for AI agent execution.

        Returns a structured dict that an AI agent can follow step-by-step.
        This does NOT actually run anything — it provides the instructions.
        """
        skill = self._registry.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found")
        if not skill.enabled:
            raise ValueError(f"Skill '{name}' is disabled")

        # Track execution
        self._execution_count[name] = self._execution_count.get(name, 0) + 1
        self._log_operation("execute", name, context)

        # Build execution plan
        steps = []
        for i, step in enumerate(sorted(skill.steps, key=lambda s: s.order or i)):
            steps.append({
                "step_number": i + 1,
                "title": step.title,
                "instruction": step.instruction,
                "tools_needed": step.tools_needed,
                "expected_output": step.expected_output,
                "on_failure": step.on_failure,
            })

        return {
            "skill_name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "total_steps": len(steps),
            "permissions": skill.permissions.model_dump(),
            "steps": steps,
            "context": context or {},
            "execution_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ── Import / Export ──────────────────────────────────────────────

    def import_skill(self, yaml_content: str, source: str = "import") -> SkillSchema:
        """Import a skill from YAML string."""
        data = yaml.safe_load(yaml_content)
        if not data or not isinstance(data, dict):
            raise ValueError("Invalid YAML content")

        name = data.get("name", "")
        if name in self._registry:
            return self.update_skill(name, data)
        else:
            return self.create_skill(data)

    def export_skill(self, name: str) -> str:
        """Export a skill as YAML string."""
        skill = self._registry.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found")
        return yaml.dump(
            skill.model_dump(exclude={"content_hash", "created_at", "updated_at"}),
            default_flow_style=False, sort_keys=False, allow_unicode=True,
        )

    # ── Toggle ───────────────────────────────────────────────────────

    def toggle_skill(self, name: str, enabled: bool) -> SkillSchema:
        """Enable or disable a skill."""
        skill = self._registry.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found")
        skill.enabled = enabled
        skill.updated_at = datetime.utcnow().isoformat()
        self._save_to_disk(skill)
        self._log_operation("toggle", name, {"enabled": enabled})
        return skill

    # ── Audit ────────────────────────────────────────────────────────

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        """Get recent skill operations."""
        return self._audit_log[-limit:][::-1]

    def _log_operation(self, operation: str, skill_name: str, extra: Any = None) -> None:
        entry = {
            "id": str(uuid.uuid4())[:8],
            "operation": operation,
            "skill_name": skill_name,
            "timestamp": datetime.utcnow().isoformat(),
            "extra": extra,
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > 500:
            self._audit_log = self._audit_log[-250:]

    # ── Internal ─────────────────────────────────────────────────────

    def _save_to_disk(self, skill: SkillSchema) -> None:
        """Persist skill to YAML file."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        fp = self._skills_dir / f"{skill.name}.yaml"
        data = skill.model_dump()
        fp.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Skill system summary."""
        total = len(self._registry)
        enabled = sum(1 for s in self._registry.values() if s.enabled)
        total_steps = sum(len(s.steps) for s in self._registry.values())
        total_executions = sum(self._execution_count.values())
        return {
            "total_skills": total,
            "enabled_skills": enabled,
            "disabled_skills": total - enabled,
            "total_steps": total_steps,
            "total_executions": total_executions,
            "skills_dir": str(self._skills_dir),
        }
