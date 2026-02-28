"""
AgentVault — Explainable Audit Narratives

Auto-generate natural language summaries from audit event chains.
Designed for compliance reports, incident reviews, and executive briefings.

Example output:
    "Agent 'data-analyst' started session at 14:02. Over 12 actions,
    it read 3 data files (allowed), attempted to delete a database record
    (DENIED — policy violation), then made 4 rapid API calls triggering
    rate limiting. A drift alert was raised at 14:07. The session ended
    at 14:15 with a trust score of 35/100 (LIMITED)."
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("agentvault.narrative")


class NarrativeGenerator:
    """
    Generates human-readable narratives from audit event sequences.

    Produces:
    - Session summaries (what happened during an agent's session)
    - Incident reports (focused on denials, escalations, drift)
    - Overview reports (aggregate across all agents)
    """

    def generate_session_narrative(
        self,
        events: list[dict[str, Any]],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Generate a natural language narrative for a session.

        Args:
            events: List of audit event dicts
            agent_id: Filter to specific agent
            session_id: Filter to specific session

        Returns:
            Human-readable narrative string
        """
        # Filter events
        filtered = events
        if agent_id:
            filtered = [e for e in filtered if e.get("agent_id") == agent_id]
        if session_id:
            filtered = [e for e in filtered if e.get("session_id") == session_id]

        if not filtered:
            return "No events found for the specified session."

        # Sort by timestamp
        filtered.sort(key=lambda e: e.get("timestamp", ""))

        # Extract metadata
        agent = filtered[0].get("agent_id", "unknown")
        session = filtered[0].get("session_id", "unknown")[:8]
        total = len(filtered)

        # Count decisions
        decisions = Counter(e.get("decision", "UNKNOWN") for e in filtered)
        allowed = decisions.get("ALLOW", 0)
        denied = decisions.get("DENY", 0)
        escalated = decisions.get("ESCALATE", 0)

        # Time range
        first_time = self._parse_time(filtered[0].get("timestamp", ""))
        last_time = self._parse_time(filtered[-1].get("timestamp", ""))
        duration = self._format_duration(first_time, last_time)

        # Group actions
        actions = Counter(e.get("action_name", "unknown") for e in filtered)
        tools = Counter(e.get("tool_name", "unknown") for e in filtered)

        # Build narrative
        parts = []

        # Opening
        time_str = first_time.strftime("%H:%M") if first_time else "unknown time"
        parts.append(
            f"**Agent '{agent}'** started session `{session}` at {time_str}."
        )

        # Action summary
        if total == 1:
            parts.append(f"During this session, 1 action was performed.")
        else:
            parts.append(
                f"Over **{total} actions** ({duration}), "
                f"the agent performed the following activities:"
            )

        # Decision breakdown
        decision_parts = []
        if allowed:
            top_allowed = [
                e.get("action_name", "?")
                for e in filtered
                if e.get("decision") == "ALLOW"
            ]
            top_actions = Counter(top_allowed).most_common(3)
            action_list = ", ".join(f"`{a}`" for a, _ in top_actions)
            decision_parts.append(
                f"- **{allowed} allowed** — including {action_list}"
            )
        if denied:
            denied_actions = [
                e.get("action_name", "?")
                for e in filtered
                if e.get("decision") == "DENY"
            ]
            denied_list = ", ".join(f"`{a}`" for a in set(denied_actions))
            deny_reasons = [
                e.get("reasoning", "")
                for e in filtered
                if e.get("decision") == "DENY" and e.get("reasoning")
            ]
            reason_text = f" ({deny_reasons[0]})" if deny_reasons else ""
            decision_parts.append(
                f"- **{denied} denied** — {denied_list}{reason_text}"
            )
        if escalated:
            esc_actions = [
                e.get("action_name", "?")
                for e in filtered
                if e.get("decision") == "ESCALATE"
            ]
            esc_list = ", ".join(f"`{a}`" for a in set(esc_actions))
            decision_parts.append(
                f"- **{escalated} escalated** for human review — {esc_list}"
            )

        parts.extend(decision_parts)

        # Highlight notable events
        notable = self._find_notable_events(filtered)
        if notable:
            parts.append("")
            parts.append("**Notable events:**")
            for note in notable[:5]:
                parts.append(f"- {note}")

        # Closing
        end_time_str = last_time.strftime("%H:%M") if last_time else "unknown time"
        risk = "low"
        if denied > 0 or escalated > 0:
            risk = "elevated"
        if denied > 2 or escalated > 1:
            risk = "high"

        parts.append("")
        parts.append(
            f"Session ended at {end_time_str}. "
            f"Overall risk assessment: **{risk}**."
        )

        return "\n".join(parts)

    def generate_overview(self, events: list[dict[str, Any]]) -> str:
        """Generate an overview narrative across all agents."""
        if not events:
            return "No audit events recorded."

        # Group by agent
        by_agent = defaultdict(list)
        for e in events:
            by_agent[e.get("agent_id", "unknown")].append(e)

        total = len(events)
        agent_count = len(by_agent)
        decisions = Counter(e.get("decision", "UNKNOWN") for e in events)

        parts = []
        parts.append(f"## Security Overview")
        parts.append("")
        parts.append(
            f"**{total} total actions** across **{agent_count} agents**."
        )
        parts.append("")
        parts.append(
            f"| Metric | Count |\n|--------|-------|\n"
            f"| Allowed | {decisions.get('ALLOW', 0)} |\n"
            f"| Denied | {decisions.get('DENY', 0)} |\n"
            f"| Escalated | {decisions.get('ESCALATE', 0)} |"
        )

        # Per-agent summary
        parts.append("")
        parts.append("### Per-Agent Summary")
        for agent_id, agent_events in sorted(by_agent.items()):
            agent_decisions = Counter(e.get("decision", "") for e in agent_events)
            parts.append(
                f"- **{agent_id}**: {len(agent_events)} actions "
                f"({agent_decisions.get('ALLOW', 0)}✓ "
                f"{agent_decisions.get('DENY', 0)}✗ "
                f"{agent_decisions.get('ESCALATE', 0)}⚠)"
            )

        return "\n".join(parts)

    def _find_notable_events(self, events: list[dict[str, Any]]) -> list[str]:
        """Identify notable events in a sequence."""
        notes = []

        for i, e in enumerate(events):
            decision = e.get("decision", "")
            action = e.get("action_name", "?")
            reasoning = e.get("reasoning", "")

            # Denied actions are always notable
            if decision == "DENY":
                notes.append(
                    f"⛔ `{action}` was **denied** — {reasoning}"
                )

            # Escalations
            elif decision == "ESCALATE":
                notes.append(
                    f"⚠️ `{action}` **escalated** for human review"
                )

            # Rapid successive calls (rate limiting indicator)
            if i > 0:
                prev_time = self._parse_time(events[i-1].get("timestamp", ""))
                curr_time = self._parse_time(e.get("timestamp", ""))
                if prev_time and curr_time:
                    delta = (curr_time - prev_time).total_seconds()
                    if 0 <= delta < 1:
                        notes.append(
                            f"⚡ Rapid successive call: `{action}` "
                            f"({delta:.1f}s after previous)"
                        )

        return notes

    @staticmethod
    def _parse_time(time_str: Any) -> Optional[datetime]:
        """Parse a timestamp string to datetime."""
        if isinstance(time_str, datetime):
            return time_str
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _format_duration(start: Optional[datetime], end: Optional[datetime]) -> str:
        """Format a duration between two timestamps."""
        if not start or not end:
            return "unknown duration"
        delta = (end - start).total_seconds()
        if delta < 60:
            return f"{delta:.0f}s"
        elif delta < 3600:
            return f"{delta/60:.0f}m {delta%60:.0f}s"
        else:
            return f"{delta/3600:.0f}h {(delta%3600)/60:.0f}m"
