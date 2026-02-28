"""
Shield Command — Natural Language Policy Engine

Converts plain English security rules into executable PolicyRules
that integrate with the existing PolicyEngine and MCPGateway.

A CFO types: "Never let AI spend more than $500 without my approval"
→ Becomes an enforceable security policy instantly.

Integration points:
    - PolicyEngine: compiled rules feed directly into policy evaluation
    - MCPGateway: enforced during the 12-step security pipeline
    - ModelRegistry: uses LLM to parse English into structured rules
    - AuditChain: all NL policies are logged with their original text
"""

import json
import uuid
import time
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum

logger = logging.getLogger("shield.nl_policy")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class PolicyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"
    REQUIRE_APPROVAL = "require_approval"
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"


@dataclass
class CompiledPolicy:
    """A policy rule compiled from natural language."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    original_text: str = ""          # The English rule as written
    name: str = ""                   # Auto-generated short name
    
    # Parsed structure
    trigger_tools: list[str] = field(default_factory=list)    # Tools this applies to
    trigger_actions: list[str] = field(default_factory=list)   # Actions: "delete", "send", "transfer"
    trigger_keywords: list[str] = field(default_factory=list)  # Content keywords
    
    condition_field: str = ""         # e.g., "amount", "count", "time"
    condition_operator: str = ""      # ">", "<", "==", "contains"
    condition_value: Any = None       # e.g., 10000, "production"
    
    action: PolicyAction = PolicyAction.BLOCK
    
    # Schedule constraints
    schedule_start: str = ""          # "02:00" (UTC)
    schedule_end: str = ""            # "05:00" (UTC)
    schedule_days: list[str] = field(default_factory=list)  # ["monday", "friday"]
    
    # Metadata
    severity: str = "medium"          # low / medium / high / critical
    author: str = "system"
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    applied_count: int = 0
    blocked_count: int = 0


# ---------------------------------------------------------------------------
# NL Policy Compiler
# ---------------------------------------------------------------------------

class NLPolicyCompiler:
    """
    Compiles natural language security rules into executable policies.
    
    Two compilation modes:
        1. LLM-powered: sends the rule to an LLM for structured extraction
        2. Pattern-based: uses regex/keyword matching for common patterns
    
    Usage:
        compiler = NLPolicyCompiler(model_registry=registry)
        policy = await compiler.compile("Never allow AI to delete production database records")
        # policy.trigger_actions = ["delete"]
        # policy.trigger_keywords = ["production", "database"]
        # policy.action = PolicyAction.BLOCK
    """

    # Common patterns for pattern-based compilation
    PATTERNS = {
        "block_action": {
            "keywords": ["never", "don't", "do not", "block", "prevent", "forbid", "prohibit", "disallow"],
            "action": PolicyAction.BLOCK,
        },
        "require_approval": {
            "keywords": ["require approval", "need approval", "get approval", "human approval", "my approval", "manager approval"],
            "action": PolicyAction.REQUIRE_APPROVAL,
        },
        "escalate": {
            "keywords": ["escalate", "alert", "notify", "flag", "warn"],
            "action": PolicyAction.ESCALATE,
        },
        "rate_limit": {
            "keywords": ["limit", "rate limit", "throttle", "maximum", "at most", "no more than"],
            "action": PolicyAction.RATE_LIMIT,
        },
    }

    DANGEROUS_ACTIONS = [
        "delete", "drop", "remove", "destroy", "truncate", "purge",
        "transfer", "send", "pay", "spend", "purchase", "buy",
        "execute", "run", "deploy", "install", "modify", "update", "alter",
    ]

    SENSITIVE_TARGETS = [
        "production", "prod", "database", "server", "credentials", "keys",
        "customer", "user data", "financial", "payment", "billing",
        "admin", "root", "sudo", "privileged",
    ]

    def __init__(self, model_registry=None, db_path: str = "shield_memory.db"):
        self.model_registry = model_registry
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nl_policies (
                    id TEXT PRIMARY KEY,
                    original_text TEXT,
                    name TEXT,
                    compiled_data TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL,
                    applied_count INTEGER DEFAULT 0,
                    blocked_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nlp_enabled ON nl_policies(enabled)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def compile(self, english_rule: str, author: str = "user") -> CompiledPolicy:
        """
        Compile an English security rule into an executable policy.
        Tries LLM first, falls back to pattern matching.
        """
        english_rule = english_rule.strip()
        if not english_rule:
            raise ValueError("Empty policy rule")

        policy = None

        # Try LLM-powered compilation first
        if self.model_registry:
            try:
                policy = await self._compile_with_llm(english_rule)
            except Exception as e:
                logger.warning(f"LLM compilation failed, falling back to patterns: {e}")

        # Fallback to pattern-based compilation
        if not policy:
            policy = self._compile_with_patterns(english_rule)

        policy.original_text = english_rule
        policy.author = author

        # Save to database
        await self._save_policy(policy)

        logger.info(
            f"Compiled policy '{policy.name}': {policy.action.value} "
            f"triggers={policy.trigger_actions} keywords={policy.trigger_keywords}"
        )
        return policy

    async def _compile_with_llm(self, rule: str) -> CompiledPolicy:
        """Use LLM to parse natural language into structured policy."""
        adapter = self.model_registry.get_adapter("reasoning")
        if not adapter:
            raise ValueError("No LLM adapter available")

        response = await adapter.generate(
            system_prompt=(
                "You are a security policy compiler. Convert the English rule into structured JSON.\n"
                "Output ONLY valid JSON with these fields:\n"
                '{\n'
                '  "name": "short descriptive name",\n'
                '  "trigger_tools": ["tool names this applies to, or empty"],\n'
                '  "trigger_actions": ["verbs like delete, send, transfer"],\n'
                '  "trigger_keywords": ["important context words like production, database"],\n'
                '  "condition_field": "field to check (amount, count, time) or empty",\n'
                '  "condition_operator": "> or < or == or contains or empty",\n'
                '  "condition_value": "threshold value or empty",\n'
                '  "action": "block | escalate | require_approval | log_only | rate_limit",\n'
                '  "severity": "low | medium | high | critical",\n'
                '  "schedule_start": "HH:MM UTC or empty",\n'
                '  "schedule_end": "HH:MM UTC or empty"\n'
                "}"
            ),
            user_prompt=f"Rule: {rule}",
            temperature=0.1,
            max_tokens=300,
        )

        content = response.get("content", "")
        parsed = self._parse_json(content)

        return CompiledPolicy(
            name=parsed.get("name", rule[:40]),
            trigger_tools=parsed.get("trigger_tools", []),
            trigger_actions=parsed.get("trigger_actions", []),
            trigger_keywords=parsed.get("trigger_keywords", []),
            condition_field=parsed.get("condition_field", ""),
            condition_operator=parsed.get("condition_operator", ""),
            condition_value=parsed.get("condition_value", ""),
            action=PolicyAction(parsed.get("action", "block")),
            severity=parsed.get("severity", "medium"),
            schedule_start=parsed.get("schedule_start", ""),
            schedule_end=parsed.get("schedule_end", ""),
        )

    def _compile_with_patterns(self, rule: str) -> CompiledPolicy:
        """Pattern-based compilation for common rule structures."""
        rule_lower = rule.lower()
        policy = CompiledPolicy(name=rule[:50])

        # Detect action type
        for pattern_type, pattern_data in self.PATTERNS.items():
            if any(kw in rule_lower for kw in pattern_data["keywords"]):
                policy.action = pattern_data["action"]
                break

        # Extract dangerous actions
        for action in self.DANGEROUS_ACTIONS:
            if action in rule_lower:
                policy.trigger_actions.append(action)

        # Extract sensitive targets
        for target in self.SENSITIVE_TARGETS:
            if target in rule_lower:
                policy.trigger_keywords.append(target)

        # Extract numeric thresholds
        import re
        numbers = re.findall(r'\$?([\d,]+(?:\.\d+)?)', rule)
        if numbers:
            value = float(numbers[0].replace(",", ""))
            policy.condition_field = "amount"
            policy.condition_operator = ">"
            policy.condition_value = value

        # Extract time constraints
        time_match = re.findall(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM|UTC))', rule)
        if len(time_match) >= 2:
            policy.schedule_start = time_match[0]
            policy.schedule_end = time_match[1]

        # Set severity based on detected elements
        if any(t in policy.trigger_keywords for t in ["production", "financial", "payment", "credentials"]):
            policy.severity = "critical"
        elif policy.trigger_actions:
            policy.severity = "high"

        return policy

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(self, policy: CompiledPolicy, tool_call: dict) -> dict:
        """
        Evaluate a tool call against a compiled policy.
        Returns {"allowed": bool, "reason": str, "policy_id": str}
        """
        if not policy.enabled:
            return {"allowed": True, "reason": "policy disabled", "policy_id": policy.id}

        tool_name = tool_call.get("tool_name", "").lower()
        params = tool_call.get("parameters", {})
        param_str = json.dumps(params).lower()

        triggered = False

        # Check tool triggers
        if policy.trigger_tools and tool_name in [t.lower() for t in policy.trigger_tools]:
            triggered = True

        # Check action triggers
        if policy.trigger_actions:
            for action in policy.trigger_actions:
                if action in tool_name or action in param_str:
                    triggered = True
                    break

        # Check keyword triggers
        if policy.trigger_keywords:
            for keyword in policy.trigger_keywords:
                if keyword in param_str or keyword in tool_name:
                    triggered = True
                    break

        # Check condition
        if triggered and policy.condition_field and policy.condition_operator:
            field_value = params.get(policy.condition_field, 0)
            try:
                field_value = float(field_value)
                threshold = float(policy.condition_value)
                if policy.condition_operator == ">" and field_value <= threshold:
                    triggered = False
                elif policy.condition_operator == "<" and field_value >= threshold:
                    triggered = False
                elif policy.condition_operator == "==" and field_value != threshold:
                    triggered = False
            except (ValueError, TypeError):
                pass

        if triggered:
            policy.applied_count += 1
            if policy.action in (PolicyAction.BLOCK, PolicyAction.REQUIRE_APPROVAL):
                policy.blocked_count += 1
            return {
                "allowed": policy.action not in (PolicyAction.BLOCK, PolicyAction.REQUIRE_APPROVAL),
                "action": policy.action.value,
                "reason": f"NL Policy '{policy.name}': {policy.original_text}",
                "policy_id": policy.id,
            }

        return {"allowed": True, "reason": "no policy triggered", "policy_id": policy.id}

    # ── Persistence ───────────────────────────────────────────────

    async def _save_policy(self, policy: CompiledPolicy):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO nl_policies 
                   (id, original_text, name, compiled_data, enabled, created_at, applied_count, blocked_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (policy.id, policy.original_text, policy.name,
                 json.dumps(asdict(policy), default=str),
                 int(policy.enabled), policy.created_at,
                 policy.applied_count, policy.blocked_count),
            )

    async def list_policies(self, enabled_only: bool = True) -> list[CompiledPolicy]:
        with self._conn() as conn:
            condition = "WHERE enabled = 1" if enabled_only else ""
            rows = conn.execute(f"SELECT compiled_data FROM nl_policies {condition}").fetchall()
        policies = []
        for row in rows:
            data = json.loads(row[0])
            data["action"] = PolicyAction(data.get("action", "block"))
            policies.append(CompiledPolicy(**{k: v for k, v in data.items() if k in CompiledPolicy.__dataclass_fields__}))
        return policies

    async def disable_policy(self, policy_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("UPDATE nl_policies SET enabled = 0 WHERE id = ?", (policy_id,))
        return True

    async def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM nl_policies").fetchone()[0]
            enabled = conn.execute("SELECT COUNT(*) FROM nl_policies WHERE enabled = 1").fetchone()[0]
            total_applied = conn.execute("SELECT SUM(applied_count) FROM nl_policies").fetchone()[0] or 0
            total_blocked = conn.execute("SELECT SUM(blocked_count) FROM nl_policies").fetchone()[0] or 0
        return {
            "total_policies": total,
            "enabled_policies": enabled,
            "total_applied": total_applied,
            "total_blocked": total_blocked,
        }

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
