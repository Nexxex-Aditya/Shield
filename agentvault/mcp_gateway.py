"""
AgentVault — MCP Gateway
Acts as both an MCP Server (agents connect in) and MCP Client (proxies to downstream tools).
This is the core innovation — the security proxy layer.

Pipeline (12 steps):
    Honeypot → Injection Scan → Chain Analysis → Policy (+Semantic) →
    Reputation Gate → Drift Detection → Shadow Execution → Sandbox →
    Surveillance → Memory Firewall → Audit → Reputation Update
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import Any, Callable, Optional

import httpx

from .models import (
    AgentAction,
    Decision,
    MCPServerConfig,
    MCPToolCall,
    MCPToolInfo,
    SandboxConfig,
)
from .policy import PolicyEngine
from .audit import AuditChain
from .drift import DriftDetector
from .confidence import ConfidenceScorer
from .sandbox import ToolSandbox, SandboxViolationError
from .honeypot import HoneypotManager
from .chain_analyzer import ChainAnalyzer
from .prompt_guard import PromptGuard
from .reputation import ReputationEngine
from .memory_firewall import MemoryFirewall
from .shadow import ShadowEngine
from .surveillance import ResponseSurveillance

logger = logging.getLogger("agentvault.mcp_gateway")


class MCPGateway:
    """
    Secure MCP Gateway — the core of AgentVault.
    
    Acts as:
    - MCP Server: agents connect to us, discover tools, make calls
    - MCP Client: we proxy those calls to downstream MCP servers
    - Security Layer: every call passes through policy, drift, sandbox, audit
    
    Flow:
        Agent → MCPGateway.handle_tool_call() → Policy → Drift → Sandbox → Downstream MCP → Audit → Response
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        audit_chain: AuditChain,
        drift_detector: DriftDetector,
        confidence_scorer: ConfidenceScorer,
        sandbox_config: Optional[SandboxConfig] = None,
        honeypot: Optional[HoneypotManager] = None,
        chain_analyzer: Optional[ChainAnalyzer] = None,
        prompt_guard: Optional[PromptGuard] = None,
        reputation: Optional[ReputationEngine] = None,
        memory_firewall: Optional[MemoryFirewall] = None,
        shadow_engine: Optional[ShadowEngine] = None,
        surveillance: Optional[ResponseSurveillance] = None,
    ) -> None:
        self._policy = policy_engine
        self._audit = audit_chain
        self._drift = drift_detector
        self._confidence = confidence_scorer
        self._sandbox = ToolSandbox(sandbox_config)

        # Advanced security modules
        self._honeypot = honeypot or HoneypotManager()
        self._chain = chain_analyzer or ChainAnalyzer()
        self._guard = prompt_guard or PromptGuard()
        self._reputation = reputation or ReputationEngine()
        self._memory = memory_firewall or MemoryFirewall()

        # Platform modules
        self._shadow = shadow_engine or ShadowEngine()
        self._surveillance = surveillance or ResponseSurveillance()

        # Downstream MCP servers
        self._servers: dict[str, MCPServerConfig] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # tool_name -> info

        # Local tool handlers (registered directly, not via MCP)
        self._local_tools: dict[str, Callable] = {}
        self._local_tool_schemas: dict[str, dict] = {}

        # Event listeners for real-time dashboard
        self._event_listeners: list[Callable] = []

        # Escalation queue
        self._escalation_queue: list[dict] = []

    # ------------------------------------------------------------------
    # Server Registration
    # ------------------------------------------------------------------

    def register_server(self, config: MCPServerConfig) -> None:
        """Register a downstream MCP server."""
        self._servers[config.server_id] = config
        logger.info("Registered MCP server: %s (%s)", config.name, config.url)

    async def discover_tools(self, server_id: str) -> list[MCPToolInfo]:
        """Discover tools from a downstream MCP server."""
        server = self._servers.get(server_id)
        if not server:
            raise ValueError(f"Unknown server: {server_id}")

        try:
            headers = {}
            if server.api_key:
                headers["Authorization"] = f"Bearer {server.api_key}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{server.url}/tools/list",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            tools = []
            for tool_data in data.get("tools", []):
                tool = MCPToolInfo(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    server_id=server_id,
                    parameters_schema=tool_data.get("inputSchema", {}),
                )
                tools.append(tool)
                # Register in our tool index (prefixed with server_id)
                full_name = f"{server_id}.{tool.name}"
                self._tools[full_name] = tool
                self._tools[tool.name] = tool  # also register without prefix

            server.tools = tools
            logger.info(
                "Discovered %d tools from server '%s'", len(tools), server.name
            )
            return tools

        except Exception as e:
            logger.error("Failed to discover tools from %s: %s", server.name, e)
            return []

    async def discover_all(self) -> dict[str, list[MCPToolInfo]]:
        """Discover tools from all registered servers."""
        results = {}
        for server_id in self._servers:
            tools = await self.discover_tools(server_id)
            results[server_id] = tools
        return results

    # ------------------------------------------------------------------
    # Local Tool Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a local tool handler (not proxied to MCP server)."""
        self._local_tools[name] = handler
        self._local_tool_schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters or {},
        }
        # Also register as MCPToolInfo for unified listing
        self._tools[name] = MCPToolInfo(
            name=name,
            description=description,
            server_id="local",
            parameters_schema=parameters or {},
        )
        logger.info("Registered local tool: %s", name)

    # ------------------------------------------------------------------
    # Core: Handle Tool Call (the main security pipeline)
    # ------------------------------------------------------------------

    async def handle_tool_call(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Handle a tool call through the full security pipeline.
        
        This is the main entry point. Every tool call goes through:
        1. Policy evaluation → ALLOW / DENY / ESCALATE
        2. Drift detection → is the agent acting normally?
        3. Sandbox execution → restricted environment
        4. Output scanning → PII / sensitive data check
        5. Audit logging → tamper-proof record
        
        Returns: {
            "success": bool,
            "result": Any,
            "decision": str,
            "reasoning": str,
            "trace_id": str,
        }
        """
        # Build the action object
        action = AgentAction(
            agent_id=agent_id,
            action_name=tool_name,
            tool_name=tool_name,
            parameters=parameters,
            context=context or {},
            session_id=session_id,
        )

        trace_id = action.trace_id
        security_flags = []

        # ---------------------------------------------------------------
        # Step 1: HONEYPOT CHECK — instant quarantine
        # ---------------------------------------------------------------
        honeypot_alert = self._honeypot.check(
            tool_name, agent_id, session_id, parameters
        )
        if honeypot_alert:
            self._reputation.record_event(agent_id, "honeypot_trigger", tool_name)
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.DENY,
                reasoning=f"🍯 HONEYPOT: Agent touched canary tool '{tool_name}' — quarantined",
            )
            response = {
                "success": False, "result": None, "decision": "DENY",
                "reasoning": f"Honeypot triggered: '{tool_name}' is a canary tool",
                "trace_id": trace_id, "security_flags": ["honeypot_triggered"],
            }
            await self._notify_listeners(response)
            return response

        # Check if agent is quarantined
        if self._honeypot.is_quarantined(agent_id):
            response = {
                "success": False, "result": None, "decision": "DENY",
                "reasoning": "Agent is quarantined — all actions blocked",
                "trace_id": trace_id, "security_flags": ["quarantined"],
            }
            await self._notify_listeners(response)
            return response

        # ---------------------------------------------------------------
        # Step 2: PROMPT INJECTION SCAN
        # ---------------------------------------------------------------
        injection_alerts = self._guard.scan(agent_id, session_id, tool_name, parameters)
        if injection_alerts and self._guard.should_block(injection_alerts):
            self._reputation.record_event(agent_id, "injection_detected", tool_name)
            security_flags.append("injection_blocked")
            max_sev = max(injection_alerts, key=lambda a: a.timestamp)
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.DENY,
                reasoning=f"Prompt injection detected: {max_sev.matched_pattern} (severity: {max_sev.severity.value})",
            )
            response = {
                "success": False, "result": None, "decision": "DENY",
                "reasoning": f"Prompt injection detected in parameters",
                "trace_id": trace_id, "security_flags": security_flags,
            }
            await self._notify_listeners(response)
            return response
        elif injection_alerts:
            security_flags.append("injection_warning")

        # ---------------------------------------------------------------
        # Step 3: CHAIN ANALYSIS — forbidden sequences
        # ---------------------------------------------------------------
        chain_violation = self._chain.record_and_check(agent_id, session_id, tool_name)
        if chain_violation:
            self._reputation.record_event(agent_id, "chain_violation", tool_name)
            security_flags.append("chain_violation")
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.DENY,
                reasoning=f"Forbidden sequence: {chain_violation.reasoning}",
            )
            response = {
                "success": False, "result": None, "decision": "DENY",
                "reasoning": f"Chain violation: {chain_violation.reasoning}",
                "trace_id": trace_id, "security_flags": security_flags,
            }
            await self._notify_listeners(response)
            return response

        # ---------------------------------------------------------------
        # Step 4: REPUTATION GATE — check trust level
        # ---------------------------------------------------------------
        if self._reputation.should_block(agent_id):
            security_flags.append("untrusted_blocked")
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.DENY,
                reasoning="Agent trust level: UNTRUSTED — all actions blocked",
            )
            response = {
                "success": False, "result": None, "decision": "DENY",
                "reasoning": "Agent blocked: trust level UNTRUSTED",
                "trace_id": trace_id, "security_flags": security_flags,
            }
            await self._notify_listeners(response)
            return response

        # ---------------------------------------------------------------
        # Step 5: POLICY EVALUATION (with semantic matching)
        # ---------------------------------------------------------------
        decision = self._policy.evaluate(action)
        logger.info(
            "[%s] Policy: %s → %s (%s)",
            trace_id[:8], tool_name, decision.decision.value, decision.reasoning,
        )

        # ---------------------------------------------------------------
        # Step 6: DRIFT DETECTION
        # ---------------------------------------------------------------
        drift_alert = self._drift.record_action(agent_id, tool_name)
        drift_score = drift_alert.deviation_score if drift_alert else 0.0

        if drift_alert:
            self._reputation.record_event(agent_id, "drift_alert", tool_name)
            security_flags.append("drift_detected")
            logger.warning(
                "[%s] Drift alert: %s (score: %.1f)",
                trace_id[:8], drift_alert.alert_level.value, drift_score,
            )

        # ---------------------------------------------------------------
        # Step 7: Handle Decision
        # ---------------------------------------------------------------
        result: Any = None
        success = False

        if decision.denied:
            self._reputation.record_event(agent_id, "policy_denial", tool_name)
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.DENY, reasoning=decision.reasoning,
                drift_score=drift_score,
            )
            response = {
                "success": False, "result": None, "decision": "DENY",
                "reasoning": decision.reasoning, "trace_id": trace_id,
                "security_flags": security_flags,
            }
            await self._notify_listeners(response)
            return response

        elif decision.escalated:
            self._reputation.record_event(agent_id, "clean_action", tool_name)
            escalation = {
                "id": trace_id, "agent_id": agent_id,
                "session_id": session_id, "trace_id": trace_id,
                "action_name": tool_name, "tool_name": tool_name,
                "parameters": parameters, "reasoning": decision.reasoning,
                "status": "PENDING", "timestamp": action.timestamp.isoformat(),
            }
            self._escalation_queue.append(escalation)
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.ESCALATE, reasoning=decision.reasoning,
                drift_score=drift_score,
            )
            response = {
                "success": False, "result": None, "decision": "ESCALATE",
                "reasoning": decision.reasoning, "trace_id": trace_id,
                "escalation_id": trace_id, "security_flags": security_flags,
            }
            await self._notify_listeners(response)
            return response

        # ---------------------------------------------------------------
        # Step 7.5: SHADOW EXECUTION — pre-commit verification
        # ---------------------------------------------------------------
        if self._shadow.should_shadow(tool_name, parameters):
            shadow_result = self._shadow.evaluate(tool_name, parameters)
            security_flags.append(f"shadow_{shadow_result.verdict}")

            if shadow_result.verdict == "block":
                self._audit.log_action(
                    agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                    action_name=tool_name, tool_name=tool_name, parameters=parameters,
                    decision=Decision.DENY,
                    reasoning=f"Shadow execution blocked: impact score {shadow_result.impact_score:.2f}",
                    drift_score=drift_score,
                )
                response = {
                    "success": False, "result": None, "decision": "DENY",
                    "reasoning": f"Shadow blocked: destructive action (impact: {shadow_result.impact_score:.2f})",
                    "trace_id": trace_id, "security_flags": security_flags,
                    "shadow_result": shadow_result.model_dump(mode="json"),
                }
                await self._notify_listeners(response)
                return response

            elif shadow_result.verdict == "escalate":
                escalation = {
                    "id": trace_id, "agent_id": agent_id,
                    "session_id": session_id, "trace_id": trace_id,
                    "action_name": tool_name, "tool_name": tool_name,
                    "parameters": parameters,
                    "reasoning": f"Shadow pre-test: impact {shadow_result.impact_score:.2f}",
                    "status": "PENDING", "timestamp": action.timestamp.isoformat(),
                    "shadow_result": shadow_result.model_dump(mode="json"),
                }
                self._escalation_queue.append(escalation)
                self._audit.log_action(
                    agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                    action_name=tool_name, tool_name=tool_name, parameters=parameters,
                    decision=Decision.ESCALATE,
                    reasoning=f"Shadow escalation: impact {shadow_result.impact_score:.2f}",
                    drift_score=drift_score,
                )
                response = {
                    "success": False, "result": None, "decision": "ESCALATE",
                    "reasoning": f"Shadow pre-test requires approval (impact: {shadow_result.impact_score:.2f})",
                    "trace_id": trace_id, "escalation_id": trace_id,
                    "security_flags": security_flags,
                    "shadow_result": shadow_result.model_dump(mode="json"),
                }
                await self._notify_listeners(response)
                return response

        # ---------------------------------------------------------------
        # Step 8: ALLOWED — Execute in Sandbox
        # ---------------------------------------------------------------
        exec_start = time.monotonic()
        try:
            if tool_name in self._local_tools:
                handler = self._local_tools[tool_name]
                result = await self._sandbox.execute(handler, **parameters)
            else:
                result = await self._proxy_to_mcp(tool_name, parameters)
            success = True

        except SandboxViolationError as e:
            self._reputation.record_event(agent_id, "sandbox_violation", tool_name)
            result = {"error": str(e), "violation": e.violation.model_dump(mode="json")}
            self._audit.log_action(
                agent_id=agent_id, session_id=session_id, trace_id=trace_id,
                action_name=tool_name, tool_name=tool_name, parameters=parameters,
                decision=Decision.DENY, reasoning=f"Sandbox violation: {e.violation.detail}",
                drift_score=drift_score, result=result,
            )
            response = {
                "success": False, "result": result, "decision": "DENY",
                "reasoning": f"Sandbox violation: {e.violation.detail}",
                "trace_id": trace_id, "security_flags": security_flags + ["sandbox_violation"],
            }
            await self._notify_listeners(response)
            return response

        except Exception as e:
            logger.error("[%s] Tool execution error: %s", trace_id[:8], e)
            result = {"error": str(e)}
            success = False

        exec_latency = (time.monotonic() - exec_start) * 1000

        # ---------------------------------------------------------------
        # Step 8.5: SURVEILLANCE — monitor tool response
        # ---------------------------------------------------------------
        response_size = len(str(result)) if result else 0
        anomaly = self._surveillance.record_response(
            tool_name=tool_name,
            success=success,
            latency_ms=exec_latency,
            response_size=response_size,
            response_data=result,
        )
        if anomaly:
            security_flags.append(f"response_{anomaly.anomaly_type}")

        # ---------------------------------------------------------------
        # Step 9: MEMORY FIREWALL — check for cross-session data smuggling
        # ---------------------------------------------------------------
        memory_violation = self._memory.check(agent_id, session_id, parameters)
        if memory_violation:
            security_flags.append("memory_violation")
            logger.warning(
                "[%s] Memory firewall: cross-session data detected (%.1f%% similarity)",
                trace_id[:8], memory_violation.similarity * 100,
            )

        # ---------------------------------------------------------------
        # Step 10: AUDIT + REPUTATION UPDATE
        # ---------------------------------------------------------------
        self._reputation.record_event(agent_id, "clean_action", tool_name)
        self._audit.log_action(
            agent_id=agent_id, session_id=session_id, trace_id=trace_id,
            action_name=tool_name, tool_name=tool_name, parameters=parameters,
            decision=Decision.ALLOW, reasoning=decision.reasoning,
            drift_score=drift_score,
            result=result if isinstance(result, dict) else {"output": str(result)} if result else None,
        )

        response = {
            "success": success, "result": result, "decision": "ALLOW",
            "reasoning": decision.reasoning, "trace_id": trace_id,
            "security_flags": security_flags,
        }
        await self._notify_listeners(response)
        return response

    # ------------------------------------------------------------------
    # MCP Client: Proxy to Downstream
    # ------------------------------------------------------------------

    async def _proxy_to_mcp(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """Proxy a tool call to the appropriate downstream MCP server."""
        # Find which server has this tool
        tool_info = self._tools.get(tool_name)
        if not tool_info:
            raise ValueError(f"Unknown tool: {tool_name}")

        server = self._servers.get(tool_info.server_id)
        if not server:
            raise ValueError(
                f"Tool '{tool_name}' belongs to server '{tool_info.server_id}' "
                f"which is not registered"
            )

        # Build the MCP tool call request
        headers = {"Content-Type": "application/json"}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        payload = {
            "name": tool_name,
            "arguments": parameters,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{server.url}/tools/call",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data.get("content", data)

    # ------------------------------------------------------------------
    # Escalation Management
    # ------------------------------------------------------------------

    def get_escalations(self, status: Optional[str] = None) -> list[dict]:
        """Get escalation queue items."""
        if status:
            return [e for e in self._escalation_queue if e["status"] == status]
        return list(self._escalation_queue)

    async def resolve_escalation(
        self, escalation_id: str, approved: bool, resolved_by: str = "admin"
    ) -> Optional[dict]:
        """Approve or reject an escalation. If approved, execute the action."""
        for esc in self._escalation_queue:
            if esc["id"] == escalation_id and esc["status"] == "PENDING":
                if approved:
                    esc["status"] = "APPROVED"
                    # Execute the previously escalated action
                    try:
                        tool_name = esc["tool_name"]
                        if tool_name in self._local_tools:
                            handler = self._local_tools[tool_name]
                            result = await self._sandbox.execute(
                                handler, **esc["parameters"]
                            )
                        else:
                            result = await self._proxy_to_mcp(
                                tool_name, esc["parameters"]
                            )
                        esc["result"] = result
                    except Exception as e:
                        esc["result"] = {"error": str(e)}
                else:
                    esc["status"] = "REJECTED"

                esc["resolved_by"] = resolved_by

                # Log the resolution
                self._audit.log_action(
                    agent_id=esc["agent_id"],
                    session_id=esc["session_id"],
                    trace_id=esc["trace_id"],
                    action_name=esc["action_name"],
                    tool_name=esc["tool_name"],
                    parameters=esc.get("parameters", {}),
                    decision=Decision.ALLOW if approved else Decision.DENY,
                    reasoning=f"Escalation {'approved' if approved else 'rejected'} by {resolved_by}",
                    result=esc.get("result"),
                )

                return esc

        return None

    # ------------------------------------------------------------------
    # Tool Listing (MCP Server interface)
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """List all available tools (from all servers + local)."""
        tools = []
        for name, info in self._tools.items():
            # Skip duplicate prefixed entries
            if "." in name:
                continue
            tools.append({
                "name": info.name,
                "description": info.description,
                "server": info.server_id,
                "parameters": info.parameters_schema,
            })
        return tools

    def list_servers(self) -> list[dict]:
        """List all registered MCP servers."""
        return [
            {
                "server_id": s.server_id,
                "name": s.name,
                "url": s.url,
                "enabled": s.enabled,
                "tool_count": len(s.tools),
            }
            for s in self._servers.values()
        ]

    # ------------------------------------------------------------------
    # Event Listeners
    # ------------------------------------------------------------------

    def add_listener(self, callback: Callable) -> None:
        """Add a real-time event listener (for WebSocket dashboard)."""
        self._event_listeners.append(callback)

    async def _notify_listeners(self, event: dict) -> None:
        """Notify all event listeners."""
        for listener in self._event_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.error("Event listener error: %s", e)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get comprehensive gateway statistics."""
        audit_stats = self._audit.get_stats()
        return {
            **audit_stats,
            "drift_alerts": len(self._drift.get_alerts()),
            "pending_escalations": len(self.get_escalations("PENDING")),
            "connected_servers": len(self._servers),
            "available_tools": len(self.list_tools()),
            "chain_healthy": self._audit.verify()[0],
            # Advanced module stats
            "honeypot_triggers": len(self._honeypot.get_triggers()),
            "quarantined_agents": len(self._honeypot.get_quarantined_agents()),
            "chain_violations": len(self._chain.get_violations()),
            "injection_alerts": len(self._guard.get_alerts()),
            "memory_violations": len(self._memory.get_violations()),
            "agent_reputations": [
                {
                    "agent_id": s.agent_id,
                    "score": round(s.score, 1),
                    "trust_level": s.trust_level.value,
                    "violations": s.violations,
                }
                for s in self._reputation.get_all_scores()
            ],
        }

    def get_security_status(self) -> dict:
        """Get comprehensive security overview across all modules."""
        return {
            "honeypot": {
                "active_canaries": len(self._honeypot.get_canary_tools()),
                "total_triggers": len(self._honeypot.get_triggers()),
                "quarantined_agents": self._honeypot.get_quarantined_agents(),
            },
            "chain_analysis": {
                "total_violations": len(self._chain.get_violations()),
                "recent": [
                    {"agent": v.agent_id, "pattern": v.matched_pattern, "reasoning": v.reasoning}
                    for v in self._chain.get_violations(limit=5)
                ],
            },
            "injection_guard": {
                "total_alerts": len(self._guard.get_alerts()),
                "recent": [
                    {"agent": a.agent_id, "tool": a.tool_name, "severity": a.severity.value}
                    for a in self._guard.get_alerts(limit=5)
                ],
            },
            "reputation": {
                "agents": [
                    s.model_dump(mode="json", exclude={"history"})
                    for s in self._reputation.get_all_scores()
                ],
            },
            "memory_firewall": {
                "total_violations": len(self._memory.get_violations()),
            },
            "policy": {
                "total_rules": sum(len(p.rules) for p in self._policy.policies),
                "hit_counts": self._policy.rule_hit_counts,
            },
            "chain_integrity": self._audit.verify()[0],
        }
