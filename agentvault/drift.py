"""
AgentVault — Behavioral Drift Detector
Statistical drift detection using chi-squared test for agent action patterns.
Detects when an agent starts doing unusual things compared to its baseline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from .models import AlertLevel, DriftAlert

logger = logging.getLogger("agentvault.drift")


class DriftDetector:
    """
    Detects behavioral drift in AI agents by comparing current action patterns
    against historical baselines using chi-squared statistical test.
    
    How it works:
    1. Build a baseline: track what actions each agent normally does and how often
    2. Monitor: as new actions come in, compare the current window against baseline
    3. Alert: if the deviation is statistically significant, fire an alert
    """

    # Alert thresholds (chi-squared deviation scores)
    THRESHOLDS = {
        AlertLevel.LOW: 5.0,
        AlertLevel.MEDIUM: 10.0,
        AlertLevel.HIGH: 20.0,
        AlertLevel.CRITICAL: 50.0,
    }

    # Time windows in seconds
    WINDOWS = {
        "1h": 3600,
        "24h": 86400,
        "7d": 604800,
    }

    def __init__(
        self,
        baseline_min_events: int = 20,
        default_window: str = "1h",
    ) -> None:
        self._baseline_min_events = baseline_min_events
        self._default_window = default_window

        # Baseline storage: agent_id -> action -> count
        self._baselines: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._baseline_totals: dict[str, int] = defaultdict(int)

        # Current event tracking: agent_id -> list of (timestamp, action)
        self._events: dict[str, list[tuple[float, str]]] = defaultdict(list)

        # Alert history
        self._alerts: list[DriftAlert] = []

        self._lock = threading.Lock()

    def record_action(self, agent_id: str, action_name: str) -> Optional[DriftAlert]:
        """
        Record an action and check for drift.
        Returns a DriftAlert if drift is detected, None otherwise.
        """
        now = time.time()

        with self._lock:
            # Add to event stream
            self._events[agent_id].append((now, action_name))

            # Update baseline
            self._baselines[agent_id][action_name] += 1
            self._baseline_totals[agent_id] += 1

            # Prune old events (keep last 7 days max)
            cutoff = now - self.WINDOWS["7d"]
            self._events[agent_id] = [
                (t, a) for t, a in self._events[agent_id] if t > cutoff
            ]

        # Check for drift if we have enough baseline data
        if self._baseline_totals.get(agent_id, 0) >= self._baseline_min_events:
            return self.check(agent_id, self._default_window)

        return None

    def check(self, agent_id: str, window: str = "1h") -> Optional[DriftAlert]:
        """
        Check if an agent's recent behavior deviates from its baseline.
        Returns a DriftAlert if drift detected, None otherwise.
        """
        window_seconds = self.WINDOWS.get(window, self.WINDOWS["1h"])
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            baseline = dict(self._baselines.get(agent_id, {}))
            baseline_total = self._baseline_totals.get(agent_id, 0)
            recent_events = [
                (t, a) for t, a in self._events.get(agent_id, []) if t > cutoff
            ]

        if not recent_events or baseline_total < self._baseline_min_events:
            return None

        # Compute current distribution
        current_counts: dict[str, int] = defaultdict(int)
        for _, action in recent_events:
            current_counts[action] += 1

        current_total = sum(current_counts.values())

        # Compute baseline distribution (proportions)
        baseline_dist = {
            action: count / baseline_total for action, count in baseline.items()
        }

        # Compute current distribution (proportions)
        current_dist = {
            action: count / current_total for action, count in current_counts.items()
        }

        # Chi-squared test
        deviation_score = self._chi_squared(
            current_counts, current_total, baseline_dist, baseline_total
        )

        # Determine alert level
        alert_level = self._get_alert_level(deviation_score)

        if alert_level is not None:
            alert = DriftAlert(
                agent_id=agent_id,
                deviation_score=deviation_score,
                alert_level=alert_level,
                baseline_distribution=baseline_dist,
                current_distribution=current_dist,
                window=window,
                message=self._format_message(
                    agent_id, deviation_score, alert_level,
                    baseline_dist, current_dist, window
                ),
            )

            with self._lock:
                self._alerts.append(alert)

            logger.warning(
                "Drift alert [%s] agent=%s score=%.2f: %s",
                alert_level.value, agent_id, deviation_score, alert.message,
            )

            return alert

        return None

    def _chi_squared(
        self,
        observed: dict[str, int],
        observed_total: int,
        expected_dist: dict[str, float],
        expected_total: int,
    ) -> float:
        """
        Compute chi-squared statistic comparing observed counts
        against expected distribution.
        """
        chi2 = 0.0

        # Get all unique actions
        all_actions = set(list(observed.keys()) + list(expected_dist.keys()))

        for action in all_actions:
            observed_count = observed.get(action, 0)
            expected_proportion = expected_dist.get(action, 0)

            # Expected count = total observed * baseline proportion
            expected_count = observed_total * expected_proportion

            if expected_count == 0:
                # New action not in baseline — this is suspicious
                if observed_count > 0:
                    chi2 += observed_count * 5.0  # penalty for unknown actions
            else:
                chi2 += (observed_count - expected_count) ** 2 / expected_count

        return chi2

    def _get_alert_level(self, deviation_score: float) -> Optional[AlertLevel]:
        """Determine alert level from deviation score."""
        if deviation_score >= self.THRESHOLDS[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        elif deviation_score >= self.THRESHOLDS[AlertLevel.HIGH]:
            return AlertLevel.HIGH
        elif deviation_score >= self.THRESHOLDS[AlertLevel.MEDIUM]:
            return AlertLevel.MEDIUM
        elif deviation_score >= self.THRESHOLDS[AlertLevel.LOW]:
            return AlertLevel.LOW
        return None

    def _format_message(
        self,
        agent_id: str,
        score: float,
        level: AlertLevel,
        baseline: dict[str, float],
        current: dict[str, float],
        window: str,
    ) -> str:
        """Format a human-readable drift alert message."""
        # Find the most deviant actions
        deviations = []
        for action in set(list(baseline.keys()) + list(current.keys())):
            b = baseline.get(action, 0)
            c = current.get(action, 0)
            diff = abs(c - b)
            if diff > 0.05:  # > 5% difference
                direction = "↑" if c > b else "↓"
                deviations.append(f"{action}: {b:.0%}→{c:.0%} {direction}")

        dev_str = ", ".join(deviations[:3]) if deviations else "pattern shift"
        return (
            f"Agent '{agent_id}' behavioral drift detected in {window} window. "
            f"Score: {score:.1f} ({level.value}). Changes: {dev_str}"
        )

    def get_alerts(
        self,
        agent_id: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        limit: int = 50,
    ) -> list[DriftAlert]:
        """Get drift alerts with optional filters."""
        with self._lock:
            alerts = list(self._alerts)

        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        if level:
            alerts = [a for a in alerts if a.alert_level == level]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_baseline(self, agent_id: str) -> dict[str, float]:
        """Get the baseline distribution for an agent."""
        with self._lock:
            baseline = dict(self._baselines.get(agent_id, {}))
            total = self._baseline_totals.get(agent_id, 0)

        if total == 0:
            return {}

        return {action: count / total for action, count in baseline.items()}

    def reset_baseline(self, agent_id: str) -> None:
        """Reset an agent's baseline (e.g., after legitimate behavior change)."""
        with self._lock:
            self._baselines[agent_id] = defaultdict(int)
            self._baseline_totals[agent_id] = 0
            self._events[agent_id] = []
        logger.info("Reset baseline for agent '%s'", agent_id)

    def clear(self) -> None:
        """Clear all data (for testing)."""
        with self._lock:
            self._baselines.clear()
            self._baseline_totals.clear()
            self._events.clear()
            self._alerts.clear()
