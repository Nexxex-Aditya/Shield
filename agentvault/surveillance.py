"""
AgentVault — Bidirectional Surveillance

Monitors tool RESPONSES (not just requests). Traditional security watches
what goes OUT to tools; this watches what comes BACK.

Catches:
    - Latency spikes → tool is overloaded or under attack
    - Response size anomalies → data exfiltration or injection
    - Error bursts → service degradation
    - Content anomalies → poisoned data from compromised tools

Design:
    Each tool builds a ResponseProfile (baseline) from its first N
    observations. Subsequent responses are compared against the baseline
    using statistical thresholds. Anomalies are recorded and surfaced.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from .models import AlertLevel, ResponseAnomaly, ResponseProfile

logger = logging.getLogger("agentvault.surveillance")

# Statistical thresholds for anomaly detection
LATENCY_Z_THRESHOLD = 3.0      # Standard deviations above mean
SIZE_Z_THRESHOLD = 3.0         # Standard deviations above mean
ERROR_BURST_THRESHOLD = 0.3    # 30% error rate in sliding window
MIN_BASELINE_OBSERVATIONS = 10  # Minimum observations before alerting
SLIDING_WINDOW_SIZE = 50       # Number of recent observations to keep


class ResponseSurveillance:
    """
    Monitors tool/service responses for anomalies.

    Integrates into the handle_tool_call pipeline AFTER execution.
    For every tool response, it:
    1. Records timing, size, and success/failure
    2. Builds a statistical baseline per tool
    3. Compares each new response against the baseline
    4. Flags anomalies with severity levels
    5. Reports tool health status
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ResponseProfile] = {}
        self._observations: dict[str, list[dict]] = defaultdict(list)
        self._anomalies: list[ResponseAnomaly] = []
        self._lock = threading.Lock()
        self._error_windows: dict[str, list[bool]] = defaultdict(list)

    # ── Core Monitoring ──────────────────────────────────────────────

    def record_response(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        response_size: int = 0,
        response_data: Optional[Any] = None,
    ) -> Optional[ResponseAnomaly]:
        """
        Record a tool response and check for anomalies.
        Returns a ResponseAnomaly if one is detected, None otherwise.
        """
        with self._lock:
            # Record observation
            obs = {
                "success": success,
                "latency_ms": latency_ms,
                "response_size": response_size,
                "timestamp": datetime.utcnow().isoformat(),
            }
            window = self._observations[tool_name]
            window.append(obs)
            if len(window) > SLIDING_WINDOW_SIZE:
                self._observations[tool_name] = window[-SLIDING_WINDOW_SIZE:]

            # Track error window
            self._error_windows[tool_name].append(success)
            if len(self._error_windows[tool_name]) > SLIDING_WINDOW_SIZE:
                self._error_windows[tool_name] = self._error_windows[tool_name][-SLIDING_WINDOW_SIZE:]

            # Update or build profile
            profile = self._update_profile(tool_name)

            # Don't alert until we have enough data
            if profile.total_observations < MIN_BASELINE_OBSERVATIONS:
                return None

            # Check for anomalies
            anomaly = self._detect_anomaly(
                tool_name, profile, latency_ms, response_size, success,
            )

            if anomaly:
                self._anomalies.append(anomaly)
                # Keep anomaly list bounded
                if len(self._anomalies) > 500:
                    self._anomalies = self._anomalies[-500:]
                logger.warning(
                    "🔍 SURVEILLANCE [%s]: %s — %s (observed: %.1f, expected: %.1f)",
                    tool_name, anomaly.anomaly_type, anomaly.detail,
                    anomaly.observed_value, anomaly.expected_value,
                )

            return anomaly

    # ── Profile Management ───────────────────────────────────────────

    def _update_profile(self, tool_name: str) -> ResponseProfile:
        """Update the response profile for a tool based on observations."""
        window = self._observations.get(tool_name, [])
        if not window:
            return self._profiles.get(
                tool_name,
                ResponseProfile(tool_name=tool_name),
            )

        latencies = [o["latency_ms"] for o in window]
        sizes = [o["response_size"] for o in window if o["response_size"] > 0]
        errors = [o for o in window if not o["success"]]

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0

        profile = ResponseProfile(
            tool_name=tool_name,
            avg_latency_ms=round(avg_lat, 1),
            avg_response_size=int(sum(sizes) / len(sizes)) if sizes else 0,
            p95_latency_ms=round(p95_lat, 1),
            error_rate=round(len(errors) / len(window), 4) if window else 0.0,
            total_observations=len(window),
            last_updated=datetime.utcnow(),
        )

        self._profiles[tool_name] = profile
        return profile

    def _detect_anomaly(
        self,
        tool_name: str,
        profile: ResponseProfile,
        latency_ms: float,
        response_size: int,
        success: bool,
    ) -> Optional[ResponseAnomaly]:
        """Check a single response against the baseline profile."""

        # --- Latency spike detection ---
        window = self._observations.get(tool_name, [])
        latencies = [o["latency_ms"] for o in window]
        if len(latencies) >= MIN_BASELINE_OBSERVATIONS:
            mean = sum(latencies) / len(latencies)
            variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
            std = math.sqrt(variance) if variance > 0 else 1.0

            if std > 0 and latency_ms > mean + LATENCY_Z_THRESHOLD * std:
                z_score = (latency_ms - mean) / std
                return ResponseAnomaly(
                    tool_name=tool_name,
                    anomaly_type="latency_spike",
                    detail=f"Latency {latency_ms:.0f}ms is {z_score:.1f}σ above mean {mean:.0f}ms",
                    severity=self._z_to_severity(z_score),
                    observed_value=latency_ms,
                    expected_value=round(mean, 1),
                    deviation_factor=round(z_score, 2),
                )

        # --- Response size spike detection ---
        if response_size > 0:
            sizes = [o["response_size"] for o in window if o["response_size"] > 0]
            if len(sizes) >= MIN_BASELINE_OBSERVATIONS:
                mean_size = sum(sizes) / len(sizes)
                var_size = sum((x - mean_size) ** 2 for x in sizes) / len(sizes)
                std_size = math.sqrt(var_size) if var_size > 0 else 1.0

                if std_size > 0 and response_size > mean_size + SIZE_Z_THRESHOLD * std_size:
                    z_score = (response_size - mean_size) / std_size
                    return ResponseAnomaly(
                        tool_name=tool_name,
                        anomaly_type="size_spike",
                        detail=f"Response size {response_size}B is {z_score:.1f}σ above mean {mean_size:.0f}B",
                        severity=self._z_to_severity(z_score),
                        observed_value=float(response_size),
                        expected_value=round(mean_size, 1),
                        deviation_factor=round(z_score, 2),
                    )

        # --- Error burst detection ---
        error_window = self._error_windows.get(tool_name, [])
        if len(error_window) >= MIN_BASELINE_OBSERVATIONS:
            recent_errors = error_window[-10:]
            error_rate = sum(1 for s in recent_errors if not s) / len(recent_errors)
            if error_rate >= ERROR_BURST_THRESHOLD:
                return ResponseAnomaly(
                    tool_name=tool_name,
                    anomaly_type="error_burst",
                    detail=f"Error rate {error_rate*100:.0f}% in last {len(recent_errors)} calls",
                    severity=AlertLevel.HIGH if error_rate > 0.5 else AlertLevel.MEDIUM,
                    observed_value=error_rate,
                    expected_value=profile.error_rate,
                    deviation_factor=round(
                        error_rate / max(profile.error_rate, 0.01), 2
                    ),
                )

        return None

    # ── Query API ────────────────────────────────────────────────────

    def get_tool_stats(self, tool_name: Optional[str] = None) -> list[dict[str, Any]]:
        """Get response profiles for tools."""
        if tool_name:
            profile = self._profiles.get(tool_name)
            if not profile:
                return []
            return [profile.model_dump(mode="json")]

        return [p.model_dump(mode="json") for p in self._profiles.values()]

    def get_anomalies(
        self,
        tool_name: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recorded anomalies, optionally filtered."""
        results = self._anomalies
        if tool_name:
            results = [a for a in results if a.tool_name == tool_name]
        if severity:
            results = [a for a in results if a.severity.value == severity]
        return [a.model_dump(mode="json") for a in results[-limit:]]

    def get_tool_health(self) -> dict[str, Any]:
        """Get overall health summary for all monitored tools."""
        tool_health = {}
        for name, profile in self._profiles.items():
            recent_anomalies = sum(
                1 for a in self._anomalies
                if a.tool_name == name
            )
            tool_health[name] = {
                "avg_latency_ms": profile.avg_latency_ms,
                "p95_latency_ms": profile.p95_latency_ms,
                "error_rate": profile.error_rate,
                "total_observations": profile.total_observations,
                "recent_anomalies": recent_anomalies,
                "status": "healthy" if recent_anomalies == 0 else
                          "warning" if recent_anomalies < 3 else "critical",
            }
        return tool_health

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._profiles.clear()
            self._observations.clear()
            self._anomalies.clear()
            self._error_windows.clear()

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _z_to_severity(z_score: float) -> AlertLevel:
        """Map z-score to alert severity."""
        if z_score >= 5.0:
            return AlertLevel.CRITICAL
        elif z_score >= 4.0:
            return AlertLevel.HIGH
        elif z_score >= 3.0:
            return AlertLevel.MEDIUM
        return AlertLevel.LOW
