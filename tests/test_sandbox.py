"""
Tests for AgentVault Tool Sandbox.
"""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.sandbox import ToolSandbox, SandboxViolationError
from agentvault.models import SandboxConfig


@pytest.fixture
def sandbox():
    config = SandboxConfig(
        allowed_paths=["/data", "/tmp"],
        allowed_domains=["api.example.com", "localhost"],
        max_execution_time=5,
        scan_pii=True,
        read_only=False,
    )
    return ToolSandbox(config)


@pytest.fixture
def readonly_sandbox():
    config = SandboxConfig(
        allowed_paths=["/data"],
        read_only=True,
    )
    return ToolSandbox(config)


class TestPathValidation:
    def test_allowed_path(self, sandbox):
        """Paths within allowed directories should pass."""
        sandbox.validate_path("/data/file.csv")
        sandbox.validate_path("/tmp/output.txt")

    def test_blocked_path(self, sandbox):
        """Paths outside allowed directories should be blocked."""
        with pytest.raises(SandboxViolationError) as exc_info:
            sandbox.validate_path("/etc/passwd")
        assert exc_info.value.violation.violation_type == "path"

    def test_no_restrictions(self):
        """Empty allowed_paths should allow everything."""
        sandbox = ToolSandbox(SandboxConfig())
        sandbox.validate_path("/any/path")  # Should not raise


class TestDomainValidation:
    def test_allowed_domain(self, sandbox):
        sandbox.validate_domain("https://api.example.com/v1/data")

    def test_subdomain(self, sandbox):
        sandbox.validate_domain("https://v2.api.example.com/resource")

    def test_blocked_domain(self, sandbox):
        with pytest.raises(SandboxViolationError) as exc_info:
            sandbox.validate_domain("https://evil.com/steal")
        assert exc_info.value.violation.violation_type == "domain"

    def test_no_restrictions(self):
        sandbox = ToolSandbox(SandboxConfig())
        sandbox.validate_domain("https://anything.com")  # Should not raise


class TestReadOnlyMode:
    def test_read_only_blocks_writes(self, readonly_sandbox):
        with pytest.raises(SandboxViolationError) as exc_info:
            readonly_sandbox.validate_write_permission()
        assert exc_info.value.violation.violation_type == "read_only"

    def test_writable_allows_writes(self, sandbox):
        sandbox.validate_write_permission()  # Should not raise


class TestPIIScanning:
    def test_ssn_detection(self, sandbox):
        text = "Patient SSN: 123-45-6789"
        cleaned, findings = sandbox.scan_output(text)
        assert len(findings) > 0
        assert "REDACTED" in cleaned
        assert any(f["type"] == "ssn" for f in findings)

    def test_credit_card_detection(self, sandbox):
        text = "Card: 4532-1234-5678-9012"
        cleaned, findings = sandbox.scan_output(text)
        assert len(findings) > 0
        assert "REDACTED" in cleaned

    def test_email_detection(self, sandbox):
        text = "Contact: user@example.com"
        cleaned, findings = sandbox.scan_output(text)
        assert len(findings) > 0
        assert "REDACTED" in cleaned

    def test_api_key_detection(self, sandbox):
        text = "Key: sk-abc123def456ghi789jkl012mno345"
        cleaned, findings = sandbox.scan_output(text)
        assert len(findings) > 0

    def test_clean_text_passes(self, sandbox):
        text = "Revenue increased 15% to $2.4M in Q4."
        cleaned, findings = sandbox.scan_output(text)
        assert len(findings) == 0
        assert cleaned == text

    def test_pii_scan_disabled(self):
        sandbox = ToolSandbox(SandboxConfig(scan_pii=False))
        text = "SSN: 123-45-6789"
        cleaned, findings = sandbox.scan_output(text)
        assert len(findings) == 0
        assert "REDACTED" not in cleaned


class TestExecution:
    @pytest.mark.asyncio
    async def test_execute_async_function(self, sandbox):
        async def my_tool(x=1, y=2):
            return x + y

        result = await sandbox.execute(my_tool, x=3, y=4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, sandbox):
        def my_tool(x=1, y=2):
            return x + y

        result = await sandbox.execute(my_tool, x=5, y=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_timeout(self):
        config = SandboxConfig(max_execution_time=1)
        sandbox = ToolSandbox(config)

        async def slow_tool():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(SandboxViolationError) as exc_info:
            await sandbox.execute(slow_tool)
        assert exc_info.value.violation.violation_type == "timeout"


class TestViolationTracking:
    def test_violations_recorded(self, sandbox):
        assert len(sandbox.violations) == 0

        with pytest.raises(SandboxViolationError):
            sandbox.validate_path("/etc/secret")

        assert len(sandbox.violations) == 1

    def test_clear_violations(self, sandbox):
        with pytest.raises(SandboxViolationError):
            sandbox.validate_path("/etc/secret")
        sandbox.clear_violations()
        assert len(sandbox.violations) == 0


class TestScopedSandbox:
    def test_create_scoped(self, sandbox):
        scoped = sandbox.create_scoped(read_only=True)
        assert scoped.config.read_only is True
        # Original should be unchanged
        assert sandbox.config.read_only is False
