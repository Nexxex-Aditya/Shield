"""
Tests for AgentVault Audit Chain.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.audit import AuditChain
from agentvault.models import Decision


@pytest.fixture
def chain():
    return AuditChain()


def _log(chain, **overrides):
    """Helper to log an audit event with sensible defaults."""
    defaults = dict(
        agent_id="test", session_id="s1", trace_id="t1",
        action_name="test", tool_name="test", parameters={},
        decision=Decision.ALLOW, reasoning="test",
    )
    defaults.update(overrides)
    return chain.log_action(**defaults)


class TestAuditChain:
    def test_log_action(self, chain):
        event = _log(chain,
            agent_id="test", session_id="session-1", trace_id="trace-1",
            action_name="read_file", tool_name="read_file",
            parameters={"path": "/data/x.csv"}, reasoning="Matched allow rule",
        )
        assert event.agent_id == "test"
        assert event.decision == Decision.ALLOW
        assert event.event_hash != ""

    def test_chain_integrity(self, chain):
        for i in range(5):
            _log(chain, trace_id=f"trace-{i}", session_id="session-1",
                 action_name="read_file", tool_name="read_file")

        valid, break_idx = chain.verify()
        assert valid is True
        assert break_idx is None

    def test_genesis_hash(self, chain):
        event = _log(chain, reasoning="first event")
        assert event.previous_hash == "GENESIS"

    def test_hash_chain_links(self, chain):
        e1 = _log(chain, trace_id="t1", reasoning="first")
        e2 = _log(chain, trace_id="t2", decision=Decision.DENY, reasoning="second")
        assert e2.previous_hash == e1.event_hash

    def test_get_session(self, chain):
        for i in range(3):
            _log(chain, session_id="session-A", trace_id=f"t-{i}")
        _log(chain, session_id="session-B", trace_id="t-other", decision=Decision.DENY)

        session_events = chain.get_session("session-A")
        assert len(session_events) == 3

    def test_query(self, chain):
        _log(chain, agent_id="agent-1", trace_id="t1",
             action_name="read_file", tool_name="read_file")
        _log(chain, agent_id="agent-2", trace_id="t2",
             action_name="delete_file", tool_name="delete_file",
             decision=Decision.DENY)

        allow_events = chain.query(decision=Decision.ALLOW)
        assert len(allow_events) == 1
        assert allow_events[0].agent_id == "agent-1"

        deny_events = chain.query(decision=Decision.DENY)
        assert len(deny_events) == 1

    def test_stats(self, chain):
        _log(chain, agent_id="a1", trace_id="t1",
             action_name="read", tool_name="read")
        _log(chain, agent_id="a1", trace_id="t2",
             action_name="delete", tool_name="delete",
             decision=Decision.DENY)

        stats = chain.get_stats()
        assert stats["total"] == 2
        assert stats["allowed"] == 1
        assert stats["denied"] == 1

    def test_export_json(self, chain):
        _log(chain)

        json_str = chain.export_json()
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["agent_id"] == "test"

    def test_event_listener(self, chain):
        received = []
        chain.add_listener(lambda event: received.append(event))
        _log(chain)
        assert len(received) == 1

    def test_count(self, chain):
        assert chain.count == 0
        _log(chain)
        assert chain.count == 1
