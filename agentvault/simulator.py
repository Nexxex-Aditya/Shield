"""
AgentVault — Policy Simulator (Dry-Run)

Replay audit event history through a candidate policy to measure impact
before deploying. Answers: "If I changed my policy, which decisions
would have been different?"

Use cases:
- Test a stricter policy before deploying to production
- Quantify impact: "This policy would have blocked 15% more actions"
- Identify affected agents: "Agent X would lose access to write_file"
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .models import Decision, SimulationResult
from .policy import PolicyEngine

logger = logging.getLogger("agentvault.simulator")


class PolicySimulator:
    """
    Replays historical audit events through a candidate policy.

    Takes:
    - A candidate PolicyEngine (loaded from a new YAML)
    - A list of historical audit events (from AuditChain or DB)

    Returns:
    - Which decisions would flip (ALLOW→DENY, DENY→ALLOW, etc.)
    - Impact summary (counts per decision type)
    - List of affected agents
    """

    def simulate(
        self,
        candidate_policy: PolicyEngine,
        audit_events: list[dict[str, Any]],
    ) -> SimulationResult:
        """
        Replay audit events through a candidate policy.

        Args:
            candidate_policy: PolicyEngine loaded with candidate rules
            audit_events: List of audit events (dicts with agent_id, action_name, etc.)

        Returns:
            SimulationResult with flipped decisions and impact analysis
        """
        flipped = []
        impact = {
            "total_evaluated": 0,
            "unchanged": 0,
            "allow_to_deny": 0,
            "deny_to_allow": 0,
            "allow_to_escalate": 0,
            "deny_to_escalate": 0,
            "escalate_to_allow": 0,
            "escalate_to_deny": 0,
        }
        affected_agents = set()

        for event in audit_events:
            agent_id = event.get("agent_id", "")
            action_name = event.get("action_name", "")
            original_decision_str = event.get("decision", "")

            if not action_name:
                continue

            impact["total_evaluated"] += 1

            # Get candidate decision
            candidate_result = candidate_policy.evaluate(
                agent_id=agent_id,
                action_name=action_name,
            )
            candidate_decision = candidate_result.decision

            # Normalize original decision
            try:
                original_decision = Decision(original_decision_str.upper())
            except (ValueError, AttributeError):
                continue

            # Compare
            if original_decision != candidate_decision:
                key = f"{original_decision.value.lower()}_to_{candidate_decision.value.lower()}"
                impact[key] = impact.get(key, 0) + 1
                affected_agents.add(agent_id)

                flipped.append({
                    "agent_id": agent_id,
                    "action_name": action_name,
                    "original": original_decision.value,
                    "candidate": candidate_decision.value,
                    "tool_name": event.get("tool_name", ""),
                })
            else:
                impact["unchanged"] += 1

        result = SimulationResult(
            total_events=impact["total_evaluated"],
            flipped_decisions=flipped[:100],  # Cap to prevent huge payloads
            impact_summary=impact,
            affected_agents=sorted(affected_agents),
        )

        logger.info(
            "📊 SIMULATION: %d events evaluated, %d decisions would flip, %d agents affected",
            result.total_events, len(flipped), len(affected_agents),
        )

        return result
