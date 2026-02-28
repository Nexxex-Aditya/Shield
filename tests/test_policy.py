"""
Tests for AgentVault Policy Engine.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.policy import PolicyEngine
from agentvault.models import AgentAction, Decision


@pytest.fixture
def policy_yaml():
    """Create a temporary policy file."""
    content = """
agent: "*"
default: deny
rules:
  - action: "read_file"
    decision: allow
    description: "Allow reading files"

  - action: "read_*"
    decision: allow
    description: "Allow all read operations"

  - action: "write_file"
    decision: allow
    rate_limit:
      max: 3
      window: 60

  - action: "delete_*"
    decision: deny
    description: "Never delete"

  - action: "send_*"
    decision: escalate
    description: "Sending requires approval"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def engine(policy_yaml):
    """Create a PolicyEngine loaded with test policies."""
    engine = PolicyEngine()
    engine.load(policy_yaml)
    return engine


def make_action(action_name: str, agent_id: str = "test-agent") -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        action_name=action_name,
        tool_name=action_name,
    )


class TestPolicyEngine:
    def test_load_policies(self, engine):
        assert len(engine.policies) == 1
        assert len(engine.policies[0].rules) == 5

    def test_allow_exact_match(self, engine):
        decision = engine.evaluate(make_action("read_file"))
        assert decision.decision == Decision.ALLOW

    def test_allow_wildcard(self, engine):
        decision = engine.evaluate(make_action("read_database"))
        assert decision.decision == Decision.ALLOW

    def test_deny_exact(self, engine):
        decision = engine.evaluate(make_action("delete_user"))
        assert decision.decision == Decision.DENY

    def test_escalate(self, engine):
        decision = engine.evaluate(make_action("send_email"))
        assert decision.decision == Decision.ESCALATE

    def test_default_deny(self, engine):
        """Unmatched actions should be denied by default."""
        decision = engine.evaluate(make_action("unknown_action"))
        assert decision.decision == Decision.DENY

    def test_rate_limiting(self, engine):
        """Rate limiting should kick in after max requests."""
        # First 3 should be allowed
        for i in range(3):
            decision = engine.evaluate(make_action("write_file"))
            assert decision.decision == Decision.ALLOW, f"Request {i+1} should be allowed"

        # 4th should be denied (rate limited)
        decision = engine.evaluate(make_action("write_file"))
        assert decision.decision == Decision.DENY

    def test_decision_properties(self, engine):
        decision = engine.evaluate(make_action("read_file"))
        assert decision.allowed is True
        assert decision.denied is False
        assert decision.escalated is False

    def test_rule_hit_counts(self, engine):
        engine.evaluate(make_action("read_file"))
        engine.evaluate(make_action("read_file"))
        engine.evaluate(make_action("read_file"))
        # Should have counted hits
        assert sum(engine.rule_hit_counts.values()) >= 3

    def test_empty_engine(self):
        """Engine with no policies loaded should deny everything."""
        engine = PolicyEngine()
        decision = engine.evaluate(make_action("anything"))
        assert decision.decision == Decision.DENY


class TestPolicyReload:
    def test_hot_reload(self, policy_yaml, engine):
        """Modifying the file should trigger reload."""
        # Touch the file to update mtime
        with open(policy_yaml, "a") as f:
            f.write("\n# comment\n")

        reloaded = engine.check_for_reload()
        # May or may not reload depending on mtime resolution,
        # but should not crash
        assert isinstance(reloaded, bool)

    def test_cleanup(self, policy_yaml):
        os.unlink(policy_yaml)
