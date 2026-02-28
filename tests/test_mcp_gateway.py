"""
Tests for AgentVault MCP Gateway.
"""

import asyncio
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.mcp_gateway import MCPGateway
from agentvault.policy import PolicyEngine
from agentvault.audit import AuditChain
from agentvault.drift import DriftDetector
from agentvault.confidence import ConfidenceScorer
from agentvault.models import SandboxConfig, Decision


@pytest.fixture
def policy_yaml():
    content = """
agent: "*"
default: deny
rules:
  - action: "read_*"
    decision: allow
  - action: "write_*"
    decision: allow
  - action: "delete_*"
    decision: deny
  - action: "send_*"
    decision: escalate
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def gateway(policy_yaml):
    policy = PolicyEngine()
    policy.load(policy_yaml)
    audit = AuditChain()
    drift = DriftDetector()
    confidence = ConfidenceScorer()
    sandbox_config = SandboxConfig(scan_pii=True)

    gw = MCPGateway(
        policy_engine=policy,
        audit_chain=audit,
        drift_detector=drift,
        confidence_scorer=confidence,
        sandbox_config=sandbox_config,
    )

    # Register mock tools
    async def mock_read_file(path=""):
        return f"Contents of {path}"

    async def mock_write_file(path="", content=""):
        return f"Written to {path}"

    async def mock_delete_file(path=""):
        return f"Deleted {path}"

    async def mock_send_email(to="", subject=""):
        return f"Email sent to {to}"

    gw.register_tool("read_file", mock_read_file, "Read a file")
    gw.register_tool("write_file", mock_write_file, "Write a file")
    gw.register_tool("delete_file", mock_delete_file, "Delete a file")
    gw.register_tool("send_email", mock_send_email, "Send email")

    return gw


class TestGatewayBasic:
    @pytest.mark.asyncio
    async def test_allow_action(self, gateway):
        result = await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="read_file", parameters={"path": "/data/x.csv"},
        )
        assert result["decision"] == "ALLOW"
        assert result["success"] is True
        assert result["trace_id"]

    @pytest.mark.asyncio
    async def test_deny_action(self, gateway):
        result = await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="delete_file", parameters={"path": "/data/x.csv"},
        )
        assert result["decision"] == "DENY"
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_escalate_action(self, gateway):
        result = await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="send_email", parameters={"to": "user@corp.com"},
        )
        assert result["decision"] == "ESCALATE"
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_result_contains_trace_id(self, gateway):
        result = await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="read_file", parameters={},
        )
        assert "trace_id" in result
        assert len(result["trace_id"]) > 0


class TestGatewayAudit:
    @pytest.mark.asyncio
    async def test_actions_recorded(self, gateway):
        await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="read_file", parameters={},
        )
        await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="delete_file", parameters={},
        )

        stats = gateway._audit.get_stats()
        assert stats["total"] >= 2


class TestGatewayEscalation:
    @pytest.mark.asyncio
    async def test_escalation_queue(self, gateway):
        result = await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="send_email", parameters={"to": "x@y.com"},
        )
        assert result["decision"] == "ESCALATE"

        pending = gateway.get_escalations("PENDING")
        assert len(pending) >= 1

    @pytest.mark.asyncio
    async def test_resolve_escalation_approve(self, gateway):
        result = await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="send_email", parameters={"to": "x@y.com"},
        )
        esc_id = result.get("escalation_id")
        if esc_id:
            resolved = await gateway.resolve_escalation(esc_id, approved=True)
            assert resolved is not None
            assert resolved["status"] == "APPROVED"


class TestGatewayToolListing:
    def test_list_tools(self, gateway):
        tools = gateway.list_tools()
        assert len(tools) >= 4
        tool_names = [t["name"] for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_list_servers(self, gateway):
        servers = gateway.list_servers()
        assert isinstance(servers, list)


class TestGatewayStats:
    @pytest.mark.asyncio
    async def test_stats(self, gateway):
        await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="read_file", parameters={},
        )
        stats = gateway.get_stats()
        assert "chain_healthy" in stats
        assert stats["available_tools"] >= 4


class TestGatewayListeners:
    @pytest.mark.asyncio
    async def test_event_listener(self, gateway):
        events = []

        async def listener(event):
            events.append(event)

        gateway.add_listener(listener)

        await gateway.handle_tool_call(
            agent_id="test", session_id="s1",
            tool_name="read_file", parameters={},
        )

        assert len(events) >= 1

    def test_cleanup(self, policy_yaml):
        os.unlink(policy_yaml)
