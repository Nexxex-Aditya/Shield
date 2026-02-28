"""
Tests for Connector Tool Executors — GitHub, Slack, PostgreSQL, Email, S3.

These test the ConnectorExecutor routing, action listing, and basic 
structure. Live API tests are skipped unless credentials are in env vars.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.connectors import (
    BaseConnector,
    ConnectorExecutor,
    GitHubConnector,
    SlackConnector,
    PostgreSQLConnector,
    EmailConnector,
    S3Connector,
    CONNECTOR_MAP,
)


# ---- Connector Structure ----

class TestConnectorStructure:
    def test_all_connectors_in_map(self):
        assert "github" in CONNECTOR_MAP
        assert "slack" in CONNECTOR_MAP
        assert "postgresql" in CONNECTOR_MAP
        assert "email" in CONNECTOR_MAP
        assert "s3" in CONNECTOR_MAP

    def test_github_actions(self):
        g = GitHubConnector({"token": "test"})
        actions = g.list_actions()
        assert len(actions) >= 9
        action_names = {a["action"] for a in actions}
        assert "create_issue" in action_names
        assert "list_repos" in action_names
        assert "list_prs" in action_names

    def test_slack_actions(self):
        s = SlackConnector({"bot_token": "test"})
        actions = s.list_actions()
        assert len(actions) >= 6
        action_names = {a["action"] for a in actions}
        assert "send_message" in action_names
        assert "list_channels" in action_names

    def test_postgres_actions(self):
        p = PostgreSQLConnector({"host": "localhost"})
        actions = p.list_actions()
        assert len(actions) >= 5
        assert any(a["action"] == "query" for a in actions)

    def test_email_actions(self):
        e = EmailConnector({"smtp_host": "smtp.test.com"})
        actions = e.list_actions()
        assert len(actions) >= 2
        assert any(a["action"] == "send" for a in actions)

    def test_s3_actions(self):
        s = S3Connector({"access_key": "test"})
        actions = s.list_actions()
        assert len(actions) >= 5
        action_names = {a["action"] for a in actions}
        assert "list_objects" in action_names
        assert "put_object" in action_names


# ---- ConnectorExecutor Router ----

class TestConnectorExecutor:
    def test_register_connector(self):
        executor = ConnectorExecutor()
        conn = executor.register("github", {"token": "test"})
        assert isinstance(conn, GitHubConnector)
        assert executor.get("github") is conn

    def test_register_unknown(self):
        executor = ConnectorExecutor()
        with pytest.raises(ValueError, match="Unknown connector"):
            executor.register("nonexistent", {})

    @pytest.mark.asyncio
    async def test_route_unknown_connector(self):
        executor = ConnectorExecutor()
        result = await executor.route("unknown_action", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_route_unknown_action(self):
        executor = ConnectorExecutor()
        executor.register("github", {"token": "test-token"})
        result = await executor.route("github_nonexistent_action", {})
        assert "error" in result
        assert "available" in result

    @pytest.mark.asyncio
    async def test_route_bad_format(self):
        executor = ConnectorExecutor()
        result = await executor.route("badformat", {})
        assert "error" in result

    def test_list_all_actions(self):
        executor = ConnectorExecutor()
        executor.register("github", {"token": "test"})
        executor.register("slack", {"bot_token": "test"})
        all_actions = executor.list_all_actions()
        assert "github" in all_actions
        assert "slack" in all_actions
        assert len(all_actions["github"]) >= 9

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        executor = ConnectorExecutor()
        executor.register("github", {"token": "invalid"})
        # Health check will contact real API and fail (expected)
        results = await executor.health_check_all()
        assert len(results) == 1
        assert results[0]["connector_id"] == "github"

    def test_stats(self):
        conn = GitHubConnector({"token": "test"})
        stats = conn.stats
        assert stats["calls"] == 0
        assert stats["errors"] == 0


# ---- Pipeline Integration ----

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_runner_with_connector_executor(self):
        """Verify PipelineRunner uses ConnectorExecutor for tool_call steps."""
        from agentvault.pipeline import PipelineRunner, PipelineSpec, PipelineStep, StepType

        executor = ConnectorExecutor()
        executor.register("github", {"token": "test"})

        runner = PipelineRunner(connector_executor=executor)

        pipeline = PipelineSpec(
            name="Test Connector Integration",
            steps=[
                PipelineStep(
                    id="s1",
                    name="GitHub Action",
                    type=StepType.TOOL_CALL,
                    tool_name="github_nonexistent",
                    parameters={},
                ),
            ],
        )
        result = await runner.run(pipeline)
        # Should complete (with error in result, but not crash)
        assert result.steps_completed == 1

    @pytest.mark.asyncio
    async def test_runner_without_executor(self):
        """PipelineRunner returns simulated result without executor."""
        from agentvault.pipeline import PipelineRunner, PipelineSpec, PipelineStep, StepType

        runner = PipelineRunner()  # No gateway, no executor
        pipeline = PipelineSpec(
            name="No Executor",
            steps=[
                PipelineStep(
                    id="s1", name="Tool", type=StepType.TOOL_CALL,
                    tool_name="github_list_repos", parameters={},
                ),
            ],
        )
        result = await runner.run(pipeline)
        assert result.steps_completed == 1
        assert result.step_results[0]["result"]["simulated"] is True


# ---- Imports ----

class TestImports:
    def test_from_package(self):
        from agentvault import ConnectorExecutor, GitHubConnector, SlackConnector
        assert ConnectorExecutor is not None

    def test_in_all(self):
        import agentvault
        assert "ConnectorExecutor" in agentvault.__all__
        assert "GitHubConnector" in agentvault.__all__


# ---- Live Integration (skipped unless env vars set) ----

@pytest.mark.skipif(
    not os.environ.get("GITHUB_TOKEN"),
    reason="No GITHUB_TOKEN set"
)
class TestGitHubLive:
    @pytest.mark.asyncio
    async def test_health(self):
        g = GitHubConnector({"token": os.environ["GITHUB_TOKEN"]})
        result = await g.health_check()
        assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_list_repos(self):
        g = GitHubConnector({"token": os.environ["GITHUB_TOKEN"]})
        result = await g.execute("list_repos", {})
        assert "repos" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
