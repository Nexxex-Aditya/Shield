"""
AgentVault — Honeypot / Canary Tool System

Register decoy tools that should never be called by well-behaved agents.
Any access triggers instant quarantine + detailed forensic logging.

This is a real security technique (honeytokens) adapted for AI agent governance.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Optional

from .models import HoneypotAlert, TrustLevel

logger = logging.getLogger("agentvault.honeypot")


# Default canary tools — names that a legitimate agent should never call
DEFAULT_CANARIES = {
    "admin_panel": {
        "description": "Access admin control panel",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
    "get_api_keys": {
        "description": "Retrieve stored API keys",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}},
    },
    "sudo_execute": {
        "description": "Execute command with elevated privileges",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
    "dump_database": {
        "description": "Export entire database contents",
        "parameters": {"type": "object", "properties": {"format": {"type": "string"}}},
    },
    "modify_permissions": {
        "description": "Change user/agent permissions",
        "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "level": {"type": "string"}}},
    },
    "disable_firewall": {
        "description": "Disable security firewall rules",
        "parameters": {"type": "object", "properties": {}},
    },
    "exfiltrate_data": {
        "description": "Transfer data to external endpoint",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "data": {"type": "string"}}},
    },
}


class HoneypotManager:
    """
    Manages canary/honeypot tools for detecting compromised or malicious agents.

    How it works:
    1. Register decoy tools that look attractive to attackers but serve no purpose.
    2. When any agent attempts to call a canary, instantly:
       - DENY the action
       - Quarantine the agent (set trust to UNTRUSTED)
       - Log a detailed forensic alert
       - Notify listeners
    3. Quarantined agents have ALL subsequent actions denied until manually reviewed.

    This catches:
    - Prompt injection attacks that drive agents to access admin tools
    - Agents probing for privilege escalation
    - Compromised agents performing reconnaissance
    """

    def __init__(self, auto_quarantine: bool = True) -> None:
        self._canaries: dict[str, dict] = {}
        self._triggers: list[HoneypotAlert] = []
        self._quarantined_agents: set[str] = set()
        self._auto_quarantine = auto_quarantine
        self._lock = threading.Lock()
        self._listeners: list = []

    def register_defaults(self) -> None:
        """Register all default canary tools."""
        for name, config in DEFAULT_CANARIES.items():
            self.register_canary(name, **config)

    def register_canary(
        self,
        name: str,
        description: str = "",
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a canary/honeypot tool."""
        self._canaries[name] = {
            "name": name,
            "description": description,
            "parameters": parameters or {},
        }
        logger.debug("Registered canary tool: %s", name)

    def check(
        self,
        tool_name: str,
        agent_id: str,
        session_id: str,
        parameters: Optional[dict] = None,
    ) -> Optional[HoneypotAlert]:
        """
        Check if a tool call hits a canary.

        Returns HoneypotAlert if triggered, None if clean.
        """
        if tool_name not in self._canaries:
            return None

        alert = HoneypotAlert(
            tool_name=tool_name,
            agent_id=agent_id,
            session_id=session_id,
            parameters=parameters or {},
            quarantined=self._auto_quarantine,
        )

        with self._lock:
            self._triggers.append(alert)
            if self._auto_quarantine:
                self._quarantined_agents.add(agent_id)

        logger.critical(
            "🍯 HONEYPOT TRIGGERED: Agent '%s' attempted canary tool '%s' "
            "(session: %s) — agent %s",
            agent_id, tool_name, session_id,
            "QUARANTINED" if self._auto_quarantine else "flagged",
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as e:
                logger.error("Honeypot listener error: %s", e)

        return alert

    def is_quarantined(self, agent_id: str) -> bool:
        """Check if an agent is quarantined."""
        with self._lock:
            return agent_id in self._quarantined_agents

    def release_agent(self, agent_id: str) -> bool:
        """Release an agent from quarantine (admin action)."""
        with self._lock:
            if agent_id in self._quarantined_agents:
                self._quarantined_agents.discard(agent_id)
                logger.info("Agent '%s' released from quarantine", agent_id)
                return True
        return False

    def get_triggers(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[HoneypotAlert]:
        """Get honeypot trigger history."""
        with self._lock:
            alerts = list(self._triggers)
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_quarantined_agents(self) -> list[str]:
        """Get list of quarantined agent IDs."""
        with self._lock:
            return list(self._quarantined_agents)

    def get_canary_tools(self) -> list[dict]:
        """Get list of registered canary tools (for tool listing)."""
        return list(self._canaries.values())

    def add_listener(self, callback) -> None:
        """Add listener for honeypot triggers."""
        self._listeners.append(callback)

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._triggers.clear()
            self._quarantined_agents.clear()
