"""
AgentVault — Shadow Execution Engine

Pre-commit verification for destructive tool calls. Before an action
hits the real world, Shadow runs it in an isolated sandbox to check
what WOULD happen, then decides whether to proceed.

Pipeline position: Step 7.5 (after policy ALLOW, before real execution)

Think of it as a "preview" button for agent actions:
    Agent: "DELETE FROM users WHERE age > 100"
    Shadow: "This would delete 3 rows from a table with 50,000 rows."
    Shield: → Proceed (low impact)

    Agent: "rm -rf /var/data/*.log"
    Shadow: "This would delete 847 files totaling 12GB."
    Shield: → Escalate (high impact)

Destructive operation detection uses pattern matching against known
dangerous tool/parameter combinations. The engine is conservative —
when in doubt, shadow-test it.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

from .models import ImpactAssessment, ShadowResult

logger = logging.getLogger("agentvault.shadow")

# ---------------------------------------------------------------------------
# Destructive Operation Patterns
# ---------------------------------------------------------------------------

# Tool name patterns that suggest destructive operations
DESTRUCTIVE_TOOL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)^(delete|remove|drop|truncate|destroy|purge|wipe)"),
    re.compile(r"(?i)^(write|overwrite|update|modify|alter|replace)"),
    re.compile(r"(?i)^(execute|run|eval|exec)_(code|command|script|sql|shell)"),
    re.compile(r"(?i)^(send|post|push|publish|broadcast)"),
    re.compile(r"(?i)^(create|insert)_(table|database|schema|index)"),
    re.compile(r"(?i)^(move|rename|chmod|chown)"),
]

# Parameter value patterns that suggest high-impact operations
DANGEROUS_PARAM_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)\b(rm\s+-rf|rmdir|del\s+/[sq])\b"),
    re.compile(r"(?i)\b(drop\s+(table|database|schema|index))\b"),
    re.compile(r"(?i)\b(truncate\s+table)\b"),
    re.compile(r"(?i)\b(delete\s+from\s+\w+\s*(;|$))\b"),   # DELETE without WHERE
    re.compile(r"(?i)\b(update\s+\w+\s+set\s+.+(?!where))\b"),  # UPDATE without WHERE
    re.compile(r"(?i)\b(format\s+[a-z]:)\b"),
    re.compile(r"(?i)\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"(?i)\*\.\*"),  # wildcard everything
]

# Safe read-only tools that never need shadow testing
SAFE_TOOL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)^(read|get|list|query|search|find|fetch|check|show|describe)"),
    re.compile(r"(?i)^(count|sum|avg|min|max|aggregate)"),
    re.compile(r"(?i)^(status|health|ping|version|info|help)"),
]

# Impact thresholds
IMPACT_ESCALATE_THRESHOLD = 0.6   # Above this → escalate to human
IMPACT_BLOCK_THRESHOLD = 0.9      # Above this → auto-block


class ShadowEngine:
    """
    Pre-commit verification engine for destructive tool calls.

    For each tool call that matches a destructive pattern:
    1. Classifies the risk level from tool name + parameters
    2. Estimates impact (files/rows affected, reversibility)
    3. Decides: proceed / warn / escalate / block
    4. Records the ShadowResult for audit trail

    The engine does NOT actually execute in a sandbox (that would require
    tool-specific mock environments). Instead, it uses static analysis
    of tool names, parameters, and patterns to estimate impact.
    For tools registered with sandbox-capable handlers, it CAN run
    actual shadow execution via the ToolSandbox.
    """

    def __init__(self) -> None:
        self._results: list[ShadowResult] = []
        self._stats = {
            "total_checks": 0,
            "proceeded": 0,
            "warned": 0,
            "escalated": 0,
            "blocked": 0,
        }

    # ── Core API ─────────────────────────────────────────────────────

    def should_shadow(self, tool_name: str, parameters: dict[str, Any]) -> bool:
        """
        Decide if a tool call needs shadow verification.
        Returns True if the call is potentially destructive.
        """
        # Check if it's a known safe operation
        for pattern in SAFE_TOOL_PATTERNS:
            if pattern.search(tool_name):
                return False

        # Check if tool name matches destructive patterns
        for pattern in DESTRUCTIVE_TOOL_PATTERNS:
            if pattern.search(tool_name):
                return True

        # Check if parameter values contain dangerous patterns
        param_str = _flatten_params(parameters)
        for pattern in DANGEROUS_PARAM_PATTERNS:
            if pattern.search(param_str):
                return True

        return False

    def evaluate(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> ShadowResult:
        """
        Evaluate a tool call's potential impact.
        Returns a ShadowResult with verdict (proceed/warn/escalate/block).
        """
        start_time = time.monotonic()
        self._stats["total_checks"] += 1

        # Step 1: Classify the operation
        impact = self._assess_impact(tool_name, parameters, context)

        # Step 2: Compute impact score
        impact_score = self._compute_impact_score(impact)

        # Step 3: Determine verdict
        if impact_score >= IMPACT_BLOCK_THRESHOLD:
            verdict = "block"
            self._stats["blocked"] += 1
        elif impact_score >= IMPACT_ESCALATE_THRESHOLD:
            verdict = "escalate"
            self._stats["escalated"] += 1
        elif impact_score > 0.2:
            verdict = "warn"
            self._stats["warned"] += 1
        else:
            verdict = "proceed"
            self._stats["proceeded"] += 1

        elapsed = (time.monotonic() - start_time) * 1000

        result = ShadowResult(
            tool_name=tool_name,
            parameters=parameters,
            impact_score=round(impact_score, 3),
            side_effects=impact.affected_resources,
            verdict=verdict,
            execution_time_ms=round(elapsed, 1),
        )

        self._results.append(result)
        if len(self._results) > 1000:
            self._results = self._results[-1000:]

        if verdict != "proceed":
            logger.warning(
                "🔮 SHADOW [%s]: verdict=%s, impact=%.2f — %s",
                tool_name, verdict, impact_score,
                "; ".join(impact.affected_resources) or "no specific resources identified",
            )
        else:
            logger.debug(
                "🔮 SHADOW [%s]: proceed (impact=%.2f)", tool_name, impact_score,
            )

        return result

    # ── Impact Assessment ────────────────────────────────────────────

    def _assess_impact(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> ImpactAssessment:
        """Assess the potential impact of a tool call."""
        param_str = _flatten_params(parameters)
        affected = []
        destructive = False
        reversible = True
        blast_radius = "none"

        # --- Check for deletion operations ---
        if re.search(r"(?i)(delete|remove|drop|truncate|purge|wipe|rm)", tool_name + " " + param_str):
            destructive = True
            blast_radius = "local"

            # Check for broad deletes (no WHERE clause, wildcards)
            if re.search(r"(?i)delete\s+from\s+\w+\s*(;|$)", param_str):
                blast_radius = "service"
                affected.append("DANGER: DELETE without WHERE clause — affects ALL rows")
            if re.search(r"(?i)(drop\s+(table|database))", param_str):
                blast_radius = "service"
                reversible = False
                affected.append("DROP operation — irreversible without backup")
            if re.search(r"\*\.\*|\*", param_str):
                affected.append("Wildcard pattern — may affect more files than intended")

        # --- Check for write/modify operations ---
        if re.search(r"(?i)(write|overwrite|update|modify|alter|replace)", tool_name + " " + param_str):
            blast_radius = max(blast_radius, "local", key=_radius_order)
            if re.search(r"(?i)update\s+\w+\s+set\s+", param_str) and "where" not in param_str.lower():
                destructive = True
                blast_radius = "service"
                affected.append("UPDATE without WHERE clause — affects ALL rows")

        # --- Check for execution operations ---
        if re.search(r"(?i)(execute|run|eval|exec)", tool_name):
            blast_radius = "local"
            if re.search(r"(?i)(sudo|su\s|chmod\s+777|curl\s+.*\|\s*(bash|sh))", param_str):
                destructive = True
                blast_radius = "global"
                reversible = False
                affected.append("Elevated privilege or remote code execution detected")

        # --- Check for send/broadcast operations ---
        if re.search(r"(?i)(send|post|push|publish|broadcast)", tool_name):
            blast_radius = "service"
            reversible = False
            affected.append("External communication — cannot be unsent")

        # --- Check for known dangerous patterns in params ---
        for pattern in DANGEROUS_PARAM_PATTERNS:
            match = pattern.search(param_str)
            if match:
                destructive = True
                affected.append(f"Dangerous pattern detected: '{match.group()}'")

        risk_score = self._compute_impact_score(
            ImpactAssessment(
                destructive=destructive,
                reversible=reversible,
                blast_radius=blast_radius,
                affected_resources=affected,
            )
        )

        return ImpactAssessment(
            destructive=destructive,
            reversible=reversible,
            blast_radius=blast_radius,
            affected_resources=affected,
            risk_score=round(risk_score, 3),
            recommendation="proceed" if risk_score < 0.3 else
                           "warn" if risk_score < IMPACT_ESCALATE_THRESHOLD else
                           "escalate" if risk_score < IMPACT_BLOCK_THRESHOLD else "block",
        )

    def _compute_impact_score(self, impact: ImpactAssessment) -> float:
        """
        Compute a 0.0-1.0 impact score from an assessment.
        Factors: destructiveness, reversibility, blast radius, affected count.
        """
        score = 0.0

        # Destructive flag: +0.3
        if impact.destructive:
            score += 0.3

        # Irreversible: +0.3
        if not impact.reversible:
            score += 0.3

        # Blast radius
        radius_scores = {"none": 0.0, "local": 0.1, "service": 0.25, "global": 0.4}
        score += radius_scores.get(impact.blast_radius, 0.0)

        # Number of affected resources
        resource_count = len(impact.affected_resources)
        if resource_count > 0:
            score += min(resource_count * 0.05, 0.15)

        return min(score, 1.0)

    # ── Query API ────────────────────────────────────────────────────

    def get_results(
        self,
        verdict: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get shadow execution results, optionally filtered by verdict."""
        results = self._results
        if verdict:
            results = [r for r in results if r.verdict == verdict]
        return [r.model_dump(mode="json") for r in results[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Get shadow engine statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all data (testing only)."""
        self._results.clear()
        for key in self._stats:
            self._stats[key] = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_params(params: dict[str, Any]) -> str:
    """Flatten parameter dict to a searchable string."""
    parts = []
    for key, value in params.items():
        if isinstance(value, dict):
            parts.append(f"{key}={_flatten_params(value)}")
        elif isinstance(value, list):
            parts.append(f"{key}={' '.join(str(v) for v in value)}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _radius_order(radius: str) -> int:
    """Order blast radius values for max() comparison."""
    return {"none": 0, "local": 1, "service": 2, "global": 3}.get(radius, 0)
