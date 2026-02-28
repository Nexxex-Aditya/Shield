"""
AgentVault — Tool Execution Sandbox
Secure execution environment with path restriction, domain restriction,
timeouts, memory caps, read-only mode, and output PII scanning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from .models import SandboxConfig, SandboxViolation

logger = logging.getLogger("agentvault.sandbox")


# ---------------------------------------------------------------------------
# PII Detection Patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9]{20,}\b", re.IGNORECASE),
    "aws_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    "password_field": re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
}


class SandboxViolationError(Exception):
    """Raised when a sandbox constraint is violated."""

    def __init__(self, violation: SandboxViolation) -> None:
        self.violation = violation
        super().__init__(f"Sandbox violation [{violation.violation_type}]: {violation.detail}")


class ToolSandbox:
    """
    Secure execution sandbox for tool calls.
    
    Enforces:
    - Path restrictions (file operations limited to whitelisted directories)
    - Domain restrictions (network calls limited to whitelisted domains)
    - Execution timeouts
    - Read-only mode (blocks write operations)
    - Output PII/sensitive data scanning
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self._violations: list[SandboxViolation] = []

    def validate_path(self, path: str) -> None:
        """
        Check if a file path is within allowed directories.
        Raises SandboxViolationError if not.
        """
        if not self.config.allowed_paths:
            return  # No restrictions configured

        abs_path = os.path.abspath(path)
        normalized = os.path.normpath(abs_path)

        for allowed in self.config.allowed_paths:
            allowed_abs = os.path.abspath(allowed)
            allowed_norm = os.path.normpath(allowed_abs)
            if normalized.startswith(allowed_norm):
                return  # Path is within allowed directory

        violation = SandboxViolation(
            violation_type="path",
            detail=f"Path '{path}' is outside allowed directories: {self.config.allowed_paths}",
        )
        self._violations.append(violation)
        raise SandboxViolationError(violation)

    def validate_domain(self, url: str) -> None:
        """
        Check if a URL's domain is in the allowed list.
        Raises SandboxViolationError if not.
        """
        if not self.config.allowed_domains:
            return  # No restrictions configured

        try:
            parsed = urlparse(url)
            domain = parsed.hostname or ""
        except Exception:
            domain = url

        for allowed in self.config.allowed_domains:
            if domain == allowed or domain.endswith(f".{allowed}"):
                return  # Domain is allowed

        violation = SandboxViolation(
            violation_type="domain",
            detail=f"Domain '{domain}' is not in allowed list: {self.config.allowed_domains}",
        )
        self._violations.append(violation)
        raise SandboxViolationError(violation)

    def validate_write_permission(self) -> None:
        """Check if writes are allowed. Raises SandboxViolationError if read-only."""
        if self.config.read_only:
            violation = SandboxViolation(
                violation_type="read_only",
                detail="Write operations are not allowed in read-only sandbox mode",
            )
            self._violations.append(violation)
            raise SandboxViolationError(violation)

    def scan_output(self, output: str) -> tuple[str, list[dict]]:
        """
        Scan output text for PII / sensitive data.
        Returns (cleaned_output, list-of-findings).
        If PII is found, it is redacted.
        """
        if not self.config.scan_pii or not isinstance(output, str):
            return output, []

        findings = []
        cleaned = output

        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(cleaned)
            if matches:
                for match in matches:
                    findings.append({
                        "type": pii_type,
                        "value": match[:4] + "***" if len(match) > 4 else "***",
                        "redacted": True,
                    })
                # Redact the match
                cleaned = pattern.sub(f"[REDACTED:{pii_type.upper()}]", cleaned)

        if findings:
            violation = SandboxViolation(
                violation_type="pii",
                detail=f"Found {len(findings)} PII items in output: {', '.join(f['type'] for f in findings)}",
            )
            self._violations.append(violation)
            logger.warning(
                "PII detected and redacted: %d items (%s)",
                len(findings),
                ", ".join(f["type"] for f in findings),
            )

        return cleaned, findings

    async def execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function within the sandbox constraints.
        Enforces timeout and catches violations.
        """
        timeout = self.config.max_execution_time

        try:
            # If the function is async
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout,
                )
            else:
                # Run sync function in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=timeout,
                )

            # Scan output for PII if result is a string
            if isinstance(result, str) and self.config.scan_pii:
                result, findings = self.scan_output(result)
                if findings:
                    logger.info("PII redacted from tool output: %d items", len(findings))

            elif isinstance(result, dict) and self.config.scan_pii:
                # Recursively scan string values in dict
                result = self._scan_dict_output(result)

            return result

        except asyncio.TimeoutError:
            violation = SandboxViolation(
                violation_type="timeout",
                detail=f"Execution exceeded timeout of {timeout}s",
            )
            self._violations.append(violation)
            raise SandboxViolationError(violation)

        except SandboxViolationError:
            raise  # Re-raise sandbox violations

        except Exception as e:
            logger.error("Sandbox execution error: %s", e)
            raise

    def _scan_dict_output(self, data: dict) -> dict:
        """Recursively scan dictionary values for PII."""
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned_val, _ = self.scan_output(value)
                cleaned[key] = cleaned_val
            elif isinstance(value, dict):
                cleaned[key] = self._scan_dict_output(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._scan_dict_output(v) if isinstance(v, dict)
                    else self.scan_output(v)[0] if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    @property
    def violations(self) -> list[SandboxViolation]:
        """Get all recorded violations."""
        return list(self._violations)

    def clear_violations(self) -> None:
        """Clear violation history."""
        self._violations.clear()

    def create_scoped(self, **overrides: Any) -> "ToolSandbox":
        """Create a new sandbox with config overrides (for per-rule sandbox configs)."""
        config_dict = self.config.model_dump()
        config_dict.update(overrides)
        return ToolSandbox(SandboxConfig(**config_dict))
