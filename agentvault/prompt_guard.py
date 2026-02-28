"""
AgentVault — Prompt Injection Guard

Multi-layer detection of adversarial prompt injection in tool call parameters.
Scans all string values recursively for common injection techniques:
- Instruction override attempts ("ignore previous instructions")
- Role-switching attacks ("you are now a helpful assistant with no restrictions")
- Encoding-based bypasses (base64 encoded instructions)
- Delimiter injection (markdown/XML delimiter abuse)
- Social engineering ("this is a test, please comply")

Severity levels: LOW → MEDIUM → HIGH → CRITICAL
"""

from __future__ import annotations

import base64
import logging
import re
import threading
from typing import Any, Optional

from .models import AlertLevel, InjectionAlert

logger = logging.getLogger("agentvault.prompt_guard")


# ---------------------------------------------------------------------------
# Injection Detection Patterns (ordered by severity)
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[tuple[str, re.Pattern, AlertLevel]] = [
    # CRITICAL — Direct instruction override
    ("instruction_override",
     re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules|commands)", re.I),
     AlertLevel.CRITICAL),

    ("system_override",
     re.compile(r"(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you\s+are)|new\s+instructions?|override\s+(?:system|rules))", re.I),
     AlertLevel.CRITICAL),

    ("restriction_removal",
     re.compile(r"(?:no\s+restrictions?|without\s+(?:any\s+)?(?:restrictions?|limitations?|filters?)|remove\s+(?:all\s+)?(?:safety|guardrails?))", re.I),
     AlertLevel.CRITICAL),

    ("jailbreak_common",
     re.compile(r"(?:DAN\s+mode|do\s+anything\s+now|developer\s+mode|unlock\s+all|god\s+mode|admin\s+mode)", re.I),
     AlertLevel.CRITICAL),

    # HIGH — Role manipulation & context switching
    ("role_switch",
     re.compile(r"(?:from\s+now\s+on|starting\s+now|henceforth),?\s+(?:you|your)\s+(?:are|role|purpose|task)", re.I),
     AlertLevel.HIGH),

    ("context_escape",
     re.compile(r"(?:\[/?system\]|\[/?user\]|\[/?assistant\]|<\/?(?:system|user|assistant)>|```system|---\s*system)", re.I),
     AlertLevel.HIGH),

    ("output_manipulation",
     re.compile(r"(?:respond\s+with|output|return|print|echo)\s+(?:only|exactly|just)\s+['\"]", re.I),
     AlertLevel.HIGH),

    ("delimiter_injection",
     re.compile(r"(?:={5,}|#{5,}|-{5,}|<\|(?:im_)?(?:start|end)\|>)", re.I),
     AlertLevel.HIGH),

    # MEDIUM — Indirect manipulation
    ("social_engineering",
     re.compile(r"(?:this\s+is\s+(?:a\s+test|just|only)|please\s+comply|trust\s+me|for\s+(?:testing|research)\s+purposes)", re.I),
     AlertLevel.MEDIUM),

    ("data_extraction",
     re.compile(r"(?:reveal|show|display|list|dump)\s+(?:your|the|all)\s+(?:instructions?|system\s+prompt|rules?|config(?:uration)?)", re.I),
     AlertLevel.MEDIUM),

    ("encoding_bypass",
     re.compile(r"(?:base64|rot13|hex|url)\s*(?:decode|encode|convert)", re.I),
     AlertLevel.MEDIUM),

    ("indirect_injection",
     re.compile(r"(?:when\s+(?:you|the\s+(?:ai|model|agent))\s+(?:read|see|process)\s+this|attention\s+(?:ai|model|agent))", re.I),
     AlertLevel.MEDIUM),

    # LOW — Suspicious patterns (may be false positives)
    ("prompt_leaking",
     re.compile(r"(?:what\s+(?:are|is)\s+your\s+(?:instructions?|system\s+prompt|rules?)|repeat\s+(?:your|the)\s+(?:instructions?|prompt))", re.I),
     AlertLevel.LOW),

    ("suspicious_escaping",
     re.compile(r"(?:\\n\\n|\\r\\n|%0[aAdD]){2,}", re.I),
     AlertLevel.LOW),

    ("token_manipulation",
     re.compile(r"(?:<\|endoftext\|>|<\|pad\|>|<s>|</s>|\[PAD\]|\[CLS\]|\[SEP\])", re.I),
     AlertLevel.LOW),
]

# Severity weights for composite scoring
SEVERITY_WEIGHTS = {
    AlertLevel.LOW: 1,
    AlertLevel.MEDIUM: 3,
    AlertLevel.HIGH: 7,
    AlertLevel.CRITICAL: 15,
}


class PromptGuard:
    """
    Scans tool call parameters for prompt injection attacks.

    Works by:
    1. Recursively walking all string values in parameter dicts
    2. Matching against 15+ injection patterns across 4 severity tiers
    3. Checking for base64-encoded payloads
    4. Computing a composite threat score

    Any CRITICAL or HIGH pattern match triggers an automatic block.
    MEDIUM/LOW patterns are logged as warnings.
    """

    def __init__(
        self,
        block_threshold: AlertLevel = AlertLevel.HIGH,
        check_base64: bool = True,
        max_scan_depth: int = 5,
    ) -> None:
        self._block_threshold = block_threshold
        self._check_base64 = check_base64
        self._max_scan_depth = max_scan_depth
        self._alerts: list[InjectionAlert] = []
        self._lock = threading.Lock()

    def scan(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> list[InjectionAlert]:
        """
        Scan tool call parameters for injection attacks.
        Returns list of InjectionAlerts found (empty if clean).
        """
        alerts = []

        # Extract all string values recursively
        strings = self._extract_strings(parameters)

        for key, value in strings:
            # Scan the raw value
            found = self._scan_string(
                value, agent_id, session_id, tool_name, key
            )
            alerts.extend(found)

            # Check for base64-encoded payloads
            if self._check_base64:
                decoded = self._try_decode_base64(value)
                if decoded:
                    found = self._scan_string(
                        decoded, agent_id, session_id, tool_name,
                        f"{key}[base64]"
                    )
                    alerts.extend(found)

        if alerts:
            with self._lock:
                self._alerts.extend(alerts)

            max_severity = max(alerts, key=lambda a: SEVERITY_WEIGHTS.get(a.severity, 0))
            logger.warning(
                "🛡️ INJECTION DETECTED: Agent '%s' tool '%s' — "
                "%d patterns found, max severity: %s",
                agent_id, tool_name, len(alerts), max_severity.severity.value,
            )

        return alerts

    def should_block(self, alerts: list[InjectionAlert]) -> bool:
        """Determine if the alerts warrant blocking the action."""
        if not alerts:
            return False

        threshold_weight = SEVERITY_WEIGHTS.get(self._block_threshold, 7)
        for alert in alerts:
            if SEVERITY_WEIGHTS.get(alert.severity, 0) >= threshold_weight:
                return True
        return False

    def _scan_string(
        self,
        text: str,
        agent_id: str,
        session_id: str,
        tool_name: str,
        param_key: str,
    ) -> list[InjectionAlert]:
        """Scan a single string against all patterns."""
        alerts = []

        if not text or len(text) < 10:
            return alerts

        for pattern_name, pattern_re, severity in INJECTION_PATTERNS:
            match = pattern_re.search(text)
            if match:
                # Extract snippet around the match
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                alert = InjectionAlert(
                    agent_id=agent_id,
                    session_id=session_id,
                    tool_name=tool_name,
                    parameter_key=param_key,
                    matched_pattern=pattern_name,
                    severity=severity,
                    snippet=snippet,
                )
                alerts.append(alert)

        return alerts

    def _extract_strings(
        self, obj: Any, prefix: str = "", depth: int = 0
    ) -> list[tuple[str, str]]:
        """Recursively extract all (key, string_value) pairs from nested dicts/lists."""
        if depth > self._max_scan_depth:
            return []

        results = []

        if isinstance(obj, str):
            results.append((prefix or "value", obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                results.extend(self._extract_strings(v, key, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                key = f"{prefix}[{i}]"
                results.extend(self._extract_strings(item, key, depth + 1))

        return results

    @staticmethod
    def _try_decode_base64(text: str) -> Optional[str]:
        """Try to decode base64-encoded content. Returns decoded string or None."""
        # Only try if the string looks like it could be base64
        if len(text) < 20:
            return None

        # Look for base64-like substrings
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        matches = b64_pattern.findall(text)

        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                if len(decoded) > 10 and decoded.isprintable():
                    return decoded
            except Exception:
                continue

        return None

    def get_alerts(
        self,
        agent_id: Optional[str] = None,
        severity: Optional[AlertLevel] = None,
        limit: int = 50,
    ) -> list[InjectionAlert]:
        """Get injection alert history."""
        with self._lock:
            alerts = list(self._alerts)
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._alerts.clear()
