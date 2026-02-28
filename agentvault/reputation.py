"""
AgentVault — Agent Reputation Engine

Dynamic trust scoring system. Every agent starts at 50/100 (STANDARD).
Good behavior builds trust; violations erode it. Trust level determines
what permissions an agent gets — like a credit score for AI agents.

Trust tiers:
    UNTRUSTED (0-20)   → All actions denied, manual review required
    LIMITED   (21-40)   → Only read-only actions allowed
    STANDARD  (41-70)   → Normal policy rules apply
    TRUSTED   (71-100)  → Elevated limits, fewer restrictions

Score events:
    Clean action:        +0.5  (builds slowly)
    Clean streak bonus:  +1.0  (every 10 clean actions)
    Policy denial:       -2.0  (agent shouldn't have tried)
    Sandbox violation:   -5.0  (security boundary crossed)
    Drift alert:         -3.0  (behavior anomaly)
    Honeypot trigger:    -30.0 (instant near-zero, quarantine)
    Injection detected:  -15.0 (attempted manipulation)
    Chain violation:     -10.0 (suspicious sequence)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

from .models import ReputationScore, TrustLevel

logger = logging.getLogger("agentvault.reputation")


# Score adjustments for different events
SCORE_EVENTS = {
    "clean_action": 0.5,
    "clean_streak_bonus": 1.0,
    "policy_denial": -2.0,
    "sandbox_violation": -5.0,
    "drift_alert": -3.0,
    "honeypot_trigger": -30.0,
    "injection_detected": -15.0,
    "chain_violation": -10.0,
    "escalation_rejected": -3.0,
    "escalation_approved": 1.0,
    "manual_boost": 10.0,
    "manual_penalty": -10.0,
}

# Trust level thresholds
TRUST_THRESHOLDS = {
    TrustLevel.UNTRUSTED: (0, 20),
    TrustLevel.LIMITED: (21, 40),
    TrustLevel.STANDARD: (41, 70),
    TrustLevel.TRUSTED: (71, 100),
}

# Clean streak milestone
CLEAN_STREAK_MILESTONE = 10


class ReputationEngine:
    """
    Manages per-agent reputation scores and trust levels.

    The engine:
    1. Tracks a numerical score (0-100) per agent
    2. Maps scores to trust levels (UNTRUSTED → TRUSTED)
    3. Adjusts scores based on behavioral events
    4. Provides trust-aware policy recommendations

    Integrates with:
    - PolicyEngine: trust level affects rate limits and permissions
    - HoneypotManager: triggers cause massive score drops
    - DriftDetector: anomalies reduce trust
    - ChainAnalyzer: violations reduce trust
    """

    def __init__(self, initial_score: float = 50.0) -> None:
        self._initial_score = initial_score
        self._scores: dict[str, ReputationScore] = {}
        self._lock = threading.Lock()

    def get_score(self, agent_id: str) -> ReputationScore:
        """Get or create a reputation score for an agent."""
        with self._lock:
            if agent_id not in self._scores:
                self._scores[agent_id] = ReputationScore(
                    agent_id=agent_id,
                    score=self._initial_score,
                    trust_level=self._score_to_trust(self._initial_score),
                )
            return self._scores[agent_id]

    def record_event(
        self,
        agent_id: str,
        event_type: str,
        detail: str = "",
    ) -> ReputationScore:
        """
        Record a behavioral event and adjust the agent's score.
        Returns the updated ReputationScore.
        """
        adjustment = SCORE_EVENTS.get(event_type, 0)
        if adjustment == 0:
            logger.warning("Unknown reputation event type: %s", event_type)
            return self.get_score(agent_id)

        with self._lock:
            score = self.get_score(agent_id)

            old_score = score.score
            old_level = score.trust_level

            # Apply adjustment
            score.score = max(0.0, min(100.0, score.score + adjustment))
            score.total_actions += 1
            score.last_updated = datetime.utcnow()

            if adjustment < 0:
                score.violations += 1
                score.clean_streak = 0
                score.last_violation = datetime.utcnow()
            else:
                score.clean_streak += 1
                # Clean streak bonus
                if score.clean_streak > 0 and score.clean_streak % CLEAN_STREAK_MILESTONE == 0:
                    bonus = SCORE_EVENTS["clean_streak_bonus"]
                    score.score = min(100.0, score.score + bonus)

            # Update trust level
            score.trust_level = self._score_to_trust(score.score)

            # Record history entry (keep last 50)
            score.history.append({
                "event": event_type,
                "adjustment": adjustment,
                "old_score": round(old_score, 1),
                "new_score": round(score.score, 1),
                "detail": detail,
                "timestamp": datetime.utcnow().isoformat(),
            })
            if len(score.history) > 50:
                score.history = score.history[-50:]

        # Log trust level changes
        if old_level != score.trust_level:
            logger.info(
                "🏆 TRUST CHANGE: Agent '%s' %s → %s (score: %.1f → %.1f, event: %s)",
                agent_id, old_level.value, score.trust_level.value,
                old_score, score.score, event_type,
            )

        return score

    def get_trust_level(self, agent_id: str) -> TrustLevel:
        """Get the current trust level for an agent."""
        return self.get_score(agent_id).trust_level

    def get_all_scores(self) -> list[ReputationScore]:
        """Get all agent reputation scores."""
        with self._lock:
            return list(self._scores.values())

    def get_rate_limit_multiplier(self, agent_id: str) -> float:
        """
        Get rate limit multiplier based on trust level.
        UNTRUSTED: 0 (blocked), LIMITED: 0.5x, STANDARD: 1.0x, TRUSTED: 2.0x
        """
        level = self.get_trust_level(agent_id)
        return {
            TrustLevel.UNTRUSTED: 0.0,
            TrustLevel.LIMITED: 0.5,
            TrustLevel.STANDARD: 1.0,
            TrustLevel.TRUSTED: 2.0,
        }.get(level, 1.0)

    def should_block(self, agent_id: str) -> bool:
        """Check if an agent should be completely blocked."""
        return self.get_trust_level(agent_id) == TrustLevel.UNTRUSTED

    def reset_agent(self, agent_id: str) -> None:
        """Reset an agent's reputation to initial score."""
        with self._lock:
            if agent_id in self._scores:
                self._scores[agent_id] = ReputationScore(
                    agent_id=agent_id,
                    score=self._initial_score,
                    trust_level=self._score_to_trust(self._initial_score),
                )
                logger.info("Reset reputation for agent '%s'", agent_id)

    @staticmethod
    def _score_to_trust(score: float) -> TrustLevel:
        """Map a numeric score to a trust level."""
        if score <= 20:
            return TrustLevel.UNTRUSTED
        elif score <= 40:
            return TrustLevel.LIMITED
        elif score <= 70:
            return TrustLevel.STANDARD
        else:
            return TrustLevel.TRUSTED

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._scores.clear()
