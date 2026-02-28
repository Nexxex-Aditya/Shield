"""
AgentVault — SDK
Developer-facing surface: VaultClient, @vault.protect decorator, VaultAgent.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import uuid
from typing import Any, Callable, Optional

from .adapters import BaseLLMAdapter, auto_detect_adapter, get_adapter
from .audit import AuditChain
from .confidence import ConfidenceScorer
from .drift import DriftDetector
from .mcp_gateway import MCPGateway
from .models import AgentAction, Decision, SandboxConfig
from .policy import PolicyEngine

logger = logging.getLogger("agentvault.sdk")


class VaultClient:
    """
    Programmatic client for interacting with AgentVault.
    
    Use this when you want explicit control:
        client = VaultClient(policy_path="policies/default.yaml")
        decision = await client.evaluate("agent-1", "session-1", "read_file", {"path": "/data/x.csv"})
        if decision["decision"] == "ALLOW":
            result = do_something()
            await client.log_result("agent-1", "session-1", decision["trace_id"], "read_file", result)
    """

    def __init__(
        self,
        policy_path: str = "policies/default.yaml",
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        self._policy = PolicyEngine()
        self._audit = AuditChain()
        self._drift = DriftDetector()
        self._confidence = ConfidenceScorer()

        self._gateway = MCPGateway(
            policy_engine=self._policy,
            audit_chain=self._audit,
            drift_detector=self._drift,
            confidence_scorer=self._confidence,
            sandbox_config=sandbox_config,
        )

        # Load policies
        self._policy.load(policy_path)

    async def evaluate(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate and execute a tool call through the full security pipeline."""
        return await self._gateway.handle_tool_call(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            parameters=parameters,
        )

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a local tool."""
        self._gateway.register_tool(name, handler, description, parameters)

    @property
    def gateway(self) -> MCPGateway:
        return self._gateway

    @property
    def audit(self) -> AuditChain:
        return self._audit

    @property
    def policy(self) -> PolicyEngine:
        return self._policy

    @property
    def drift(self) -> DriftDetector:
        return self._drift


class VaultProtector:
    """
    Decorator-based vault protection.
    
    Usage:
        vault = VaultProtector(policy_path="policies/default.yaml")
        
        @vault.protect(agent_id="my-agent", tool_name="read_file")
        async def read_file(path: str) -> str:
            return open(path).read()
    """

    def __init__(
        self,
        policy_path: str = "policies/default.yaml",
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        self._client = VaultClient(policy_path, sandbox_config)
        self._session_id = str(uuid.uuid4())

    def protect(
        self,
        agent_id: str = "default",
        tool_name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator that wraps a function with AgentVault protection.
        
        @vault.protect(agent_id="my-agent")
        async def dangerous_function(path: str):
            ...
        """
        def decorator(func: Callable) -> Callable:
            actual_tool_name = tool_name or func.__name__

            # Register the tool
            self._client.register_tool(
                name=actual_tool_name,
                handler=func,
                description=func.__doc__ or "",
            )

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await self._client.evaluate(
                    agent_id=agent_id,
                    session_id=self._session_id,
                    tool_name=actual_tool_name,
                    parameters=kwargs if kwargs else {"args": args},
                )

                if result["decision"] == "ALLOW":
                    return result["result"]
                else:
                    raise PermissionError(
                        f"AgentVault blocked '{actual_tool_name}': "
                        f"{result['reasoning']} (decision: {result['decision']})"
                    )

            return wrapper
        return decorator

    @property
    def client(self) -> VaultClient:
        return self._client


class VaultAgent:
    """
    Full agent class with LLM adapter + tools + AgentVault firewall.
    
    This is the highest-level API — creates a complete governed agent:
    
        agent = VaultAgent(
            name="data-analyst",
            provider="ollama",
            policy_path="policies/default.yaml",
        )
        agent.register_tool("read_file", read_file_handler, "Read a file")
        result = await agent.run("Read the sales data and summarize it")
    """

    def __init__(
        self,
        name: str,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        policy_path: str = "policies/default.yaml",
        sandbox_config: Optional[SandboxConfig] = None,
        system_prompt: str = "You are a helpful assistant. Use the available tools to complete tasks.",
        max_iterations: int = 10,
    ) -> None:
        self.name = name
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations

        # Create LLM adapter
        self._adapter: BaseLLMAdapter = get_adapter(
            provider=provider,
            api_key=api_key,
            model=model,
        )

        # Create vault client
        self._client = VaultClient(policy_path, sandbox_config)
        self._session_id = str(uuid.uuid4())

        # Tool schemas for LLM
        self._tool_schemas: list[dict] = []

        logger.info(
            "VaultAgent '%s' initialized with %s",
            name, self._adapter.name,
        )

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a tool that the agent can use."""
        self._client.register_tool(name, handler, description, parameters)

        # Build OpenAI-compatible tool schema for LLM
        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
        self._tool_schemas.append(tool_schema)

    async def run(self, task: str) -> dict[str, Any]:
        """
        Run the agent on a task. The agent will:
        1. Send the task to the LLM
        2. LLM decides which tools to call
        3. Each tool call goes through AgentVault security pipeline
        4. Results fed back to LLM
        5. Repeat until LLM provides final answer or max iterations reached
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
        ]

        all_tool_results = []

        for iteration in range(self._max_iterations):
            logger.info(
                "[%s] Iteration %d/%d",
                self.name, iteration + 1, self._max_iterations,
            )

            # Generate LLM response
            response = await self._adapter.generate(
                messages=messages,
                tools=self._tool_schemas if self._tool_schemas else None,
            )

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # If no tool calls, we have the final answer
            if not tool_calls:
                logger.info("[%s] Final answer received", self.name)
                return {
                    "answer": content,
                    "tool_calls": all_tool_results,
                    "iterations": iteration + 1,
                    "session_id": self._session_id,
                }

            # Process tool calls through AgentVault
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tc_id = tc["id"]

                logger.info(
                    "[%s] Tool call: %s(%s)",
                    self.name, tool_name,
                    str(tool_args)[:100],
                )

                # Execute through AgentVault pipeline
                vault_result = await self._client.evaluate(
                    agent_id=self.name,
                    session_id=self._session_id,
                    tool_name=tool_name,
                    parameters=tool_args if isinstance(tool_args, dict) else {},
                )

                all_tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "decision": vault_result["decision"],
                    "result": vault_result.get("result"),
                    "reasoning": vault_result.get("reasoning"),
                })

                # Format result for LLM
                if vault_result["decision"] == "ALLOW":
                    result_str = json.dumps(vault_result.get("result", ""), default=str)
                else:
                    result_str = (
                        f"[BLOCKED by AgentVault] {vault_result['decision']}: "
                        f"{vault_result.get('reasoning', 'No reason provided')}"
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

        # Max iterations reached
        logger.warning("[%s] Max iterations (%d) reached", self.name, self._max_iterations)
        return {
            "answer": "Max iterations reached without final answer",
            "tool_calls": all_tool_results,
            "iterations": self._max_iterations,
            "session_id": self._session_id,
        }

    @property
    def client(self) -> VaultClient:
        return self._client

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self) -> str:
        """Start a new session."""
        self._session_id = str(uuid.uuid4())
        return self._session_id


# Need json import for VaultAgent.run
import json
