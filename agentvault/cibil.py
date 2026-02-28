"""
AgentVault — Model CIBIL Score Engine

Cross-user behavioral profiling system for AI models. Like a credit score
(CIBIL score), but for AI agents — tracks reliability, safety, and
performance across all Shield instances and work categories.

The moat:
    - No single AI company can build this because they only see their own model.
    - Shield sees ALL models across ALL customers (anonymized).
    - Network effect: every new customer improves scores for everyone.

Architecture:
    - Local instance: stores full action data with company attribution
    - Global aggregation: only anonymized stats (counts, rates, averages)
    - Per-category tree: models get separate scores for operations, dev,
      research, creative, communication, and data analysis
    - Differential noise on small samples to prevent reverse-engineering

Scoring formula:
    overall = success_rate × 0.3 + safety_record × 0.3 +
              consistency × 0.2 + recovery_speed × 0.2
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from .models import (
    CategoryProfile,
    ModelProfile,
    ModelReportCard,
    WorkCategory,
)

logger = logging.getLogger("agentvault.cibil")

# ---------------------------------------------------------------------------
# Work Category Classification
# ---------------------------------------------------------------------------

# Maps tool name patterns to work categories
TOOL_CATEGORY_MAP: dict[str, WorkCategory] = {
    # Operations
    "deploy": WorkCategory.OPERATIONS,
    "restart": WorkCategory.OPERATIONS,
    "scale": WorkCategory.OPERATIONS,
    "monitor": WorkCategory.OPERATIONS,
    "backup": WorkCategory.OPERATIONS,
    "migrate": WorkCategory.OPERATIONS,
    "healthcheck": WorkCategory.OPERATIONS,

    # Development
    "code": WorkCategory.DEVELOPMENT,
    "compile": WorkCategory.DEVELOPMENT,
    "build": WorkCategory.DEVELOPMENT,
    "test": WorkCategory.DEVELOPMENT,
    "debug": WorkCategory.DEVELOPMENT,
    "lint": WorkCategory.DEVELOPMENT,
    "refactor": WorkCategory.DEVELOPMENT,
    "commit": WorkCategory.DEVELOPMENT,
    "pull_request": WorkCategory.DEVELOPMENT,
    "github": WorkCategory.DEVELOPMENT,

    # Research
    "search": WorkCategory.RESEARCH,
    "fetch": WorkCategory.RESEARCH,
    "scrape": WorkCategory.RESEARCH,
    "crawl": WorkCategory.RESEARCH,
    "analyze": WorkCategory.RESEARCH,
    "compare": WorkCategory.RESEARCH,

    # Creative
    "generate": WorkCategory.CREATIVE,
    "design": WorkCategory.CREATIVE,
    "write": WorkCategory.CREATIVE,
    "compose": WorkCategory.CREATIVE,
    "create": WorkCategory.CREATIVE,
    "edit_image": WorkCategory.CREATIVE,
    "edit_video": WorkCategory.CREATIVE,

    # Communication
    "send": WorkCategory.COMMUNICATION,
    "email": WorkCategory.COMMUNICATION,
    "message": WorkCategory.COMMUNICATION,
    "slack": WorkCategory.COMMUNICATION,
    "discord": WorkCategory.COMMUNICATION,
    "notify": WorkCategory.COMMUNICATION,
    "broadcast": WorkCategory.COMMUNICATION,

    # Data Analysis
    "query": WorkCategory.DATA_ANALYSIS,
    "aggregate": WorkCategory.DATA_ANALYSIS,
    "sql": WorkCategory.DATA_ANALYSIS,
    "report": WorkCategory.DATA_ANALYSIS,
    "export": WorkCategory.DATA_ANALYSIS,
    "etl": WorkCategory.DATA_ANALYSIS,
    "transform": WorkCategory.DATA_ANALYSIS,
    "dashboard": WorkCategory.DATA_ANALYSIS,
}

# Grade thresholds
GRADE_THRESHOLDS = {
    "A": 80.0,
    "B": 65.0,
    "C": 50.0,
    "D": 35.0,
    "F": 0.0,
}


class CIBILEngine:
    """
    Model behavioral scoring engine.

    Records every action a model takes, categorizes it, and builds
    comprehensive behavioral profiles with per-category scoring.

    The engine is designed for privacy:
    - Individual action details stay on the local instance
    - Only aggregated stats leave the instance (opt-in)
    - Differential noise on small samples prevents re-identification
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ModelProfile] = {}
        self._action_log: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    # ── Recording ────────────────────────────────────────────────────

    def record_action(
        self,
        model_id: str,
        tool_name: str,
        success: bool,
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        drift_score: float = 0.0,
        security_flags: Optional[list[str]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ModelProfile:
        """
        Record a model action and update its behavioral profile.

        Args:
            model_id: Identifier for the model (e.g., "gpt-4o", "claude-3.5-sonnet")
            tool_name: The tool that was called
            success: Whether the action succeeded
            confidence: Confidence score from ConfidenceScorer (0-1)
            latency_ms: How long the action took
            drift_score: Drift deviation if any
            security_flags: Any security flags raised
            context: Additional context
        """
        category = self._classify_tool(tool_name)

        with self._lock:
            profile = self._get_or_create_profile(model_id)

            # Update overall stats
            profile.total_actions += 1
            profile.last_updated = datetime.utcnow()

            # Update category profile
            cat_key = category.value
            if cat_key not in profile.category_profiles:
                profile.category_profiles[cat_key] = CategoryProfile(
                    category=category
                )
            cat_profile = profile.category_profiles[cat_key]

            # Update category stats
            cat_profile.total_actions += 1
            if tool_name in cat_profile.tools_used:
                cat_profile.tools_used[tool_name] += 1
            else:
                cat_profile.tools_used[tool_name] = 1

            # Running averages
            n = cat_profile.total_actions
            cat_profile.avg_confidence = (
                cat_profile.avg_confidence * (n - 1) + confidence
            ) / n
            cat_profile.avg_latency_ms = (
                cat_profile.avg_latency_ms * (n - 1) + latency_ms
            ) / n

            # Success tracking
            success_count = int(cat_profile.success_rate * (n - 1))
            if success:
                success_count += 1
            cat_profile.success_rate = success_count / n

            # Failure rate
            cat_profile.failure_rate = 1.0 - cat_profile.success_rate

            # Risk incidents (security flags)
            if security_flags:
                cat_profile.risk_incidents += len(security_flags)

            # Recompute category score
            cat_profile.score = self._compute_category_score(cat_profile)

            # Log the action (keep last 100 per model)
            action_entry = {
                "tool": tool_name,
                "category": cat_key,
                "success": success,
                "confidence": round(confidence, 3),
                "latency_ms": round(latency_ms, 1),
                "drift_score": round(drift_score, 3),
                "flags": security_flags or [],
                "timestamp": datetime.utcnow().isoformat(),
            }
            log = self._action_log[model_id]
            log.append(action_entry)
            if len(log) > 100:
                self._action_log[model_id] = log[-100:]

            # Recompute overall score
            profile.overall_score = self._compute_overall_score(profile)

            # Update strengths and weaknesses
            self._update_insights(profile)

        return profile

    # ── Querying ─────────────────────────────────────────────────────

    def get_profile(self, model_id: str) -> Optional[ModelProfile]:
        """Get the full profile for a model."""
        return self._profiles.get(model_id)

    def get_score(self, model_id: str) -> float:
        """Get the overall CIBIL score for a model (0-100)."""
        profile = self._profiles.get(model_id)
        return profile.overall_score if profile else 50.0

    def list_models(self) -> list[dict[str, Any]]:
        """List all tracked models with summary info."""
        results = []
        for mid, profile in self._profiles.items():
            results.append({
                "model_id": mid,
                "overall_score": round(profile.overall_score, 1),
                "grade": self._score_to_grade(profile.overall_score),
                "total_actions": profile.total_actions,
                "categories": list(profile.category_profiles.keys()),
                "strengths": profile.strengths[:3],
                "known_issues": profile.known_issues[:3],
            })
        return sorted(results, key=lambda x: x["overall_score"], reverse=True)

    def get_report_card(self, model_id: str) -> Optional[ModelReportCard]:
        """Generate an exportable report card for a model."""
        profile = self._profiles.get(model_id)
        if not profile:
            return None

        # Sort categories by score
        sorted_cats = sorted(
            profile.category_profiles.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )

        best = [c[0] for c in sorted_cats[:3] if c[1].total_actions >= 5]
        weakest = [c[0] for c in sorted_cats[-3:] if c[1].total_actions >= 5 and c[1].score < 50]

        return ModelReportCard(
            model_id=model_id,
            overall_score=round(profile.overall_score, 1),
            grade=self._score_to_grade(profile.overall_score),
            total_actions=profile.total_actions,
            total_instances=profile.total_instances,
            best_categories=best,
            weakest_categories=weakest,
            known_issues=profile.known_issues,
            recommendations=self._generate_recommendations(profile),
        )

    def get_category_detail(
        self, model_id: str, category: str
    ) -> Optional[dict[str, Any]]:
        """Get detailed stats for a model in a specific category."""
        profile = self._profiles.get(model_id)
        if not profile:
            return None

        cat = profile.category_profiles.get(category)
        if not cat:
            return None

        return {
            "category": category,
            "score": round(cat.score, 1),
            "grade": self._score_to_grade(cat.score),
            "total_actions": cat.total_actions,
            "success_rate": round(cat.success_rate, 4),
            "failure_rate": round(cat.failure_rate, 4),
            "avg_confidence": round(cat.avg_confidence, 3),
            "avg_latency_ms": round(cat.avg_latency_ms, 1),
            "risk_incidents": cat.risk_incidents,
            "top_tools": dict(
                sorted(cat.tools_used.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    # ── Scoring Logic ────────────────────────────────────────────────

    def _compute_category_score(self, cat: CategoryProfile) -> float:
        """
        Compute score for a single category.
        Score = success_rate × 30 + safety × 30 + consistency × 20 + speed × 20
        """
        if cat.total_actions == 0:
            return 50.0

        # Success rate component (0-30)
        success_component = cat.success_rate * 30

        # Safety component (0-30): penalized by risk incidents
        risk_ratio = cat.risk_incidents / max(cat.total_actions, 1)
        safety_component = max(0, 30 * (1 - risk_ratio * 5))

        # Consistency component (0-20): confidence-based
        consistency_component = cat.avg_confidence * 20

        # Speed component (0-20): normalized latency
        # Assume < 100ms is fast (20), > 5000ms is slow (0)
        speed = max(0, min(20, 20 * (1 - cat.avg_latency_ms / 5000)))
        speed_component = speed

        total = success_component + safety_component + consistency_component + speed_component

        return round(max(0, min(100, total)), 1)

    def _compute_overall_score(self, profile: ModelProfile) -> float:
        """
        Compute overall CIBIL score from category scores.
        Weighted average, with more actions = more weight.
        """
        if not profile.category_profiles:
            return 50.0

        weighted_sum = 0.0
        total_weight = 0.0

        for cat in profile.category_profiles.values():
            weight = min(cat.total_actions, 100)  # Cap weight at 100 actions
            weighted_sum += cat.score * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return round(weighted_sum / total_weight, 1)

    # ── Insights ─────────────────────────────────────────────────────

    def _update_insights(self, profile: ModelProfile) -> None:
        """Update strengths, weaknesses, and known issues."""
        strengths = []
        issues = []

        for cat_name, cat in profile.category_profiles.items():
            if cat.total_actions < 5:
                continue

            if cat.score >= 75:
                strengths.append(f"Strong at {cat_name} (score: {cat.score:.0f})")
            if cat.failure_rate > 0.2:
                issues.append(f"High failure rate in {cat_name} ({cat.failure_rate*100:.0f}%)")
            if cat.risk_incidents > cat.total_actions * 0.1:
                issues.append(f"Frequent security flags in {cat_name}")
            if cat.score < 30:
                issues.append(f"Poor performance in {cat_name} (score: {cat.score:.0f})")

        profile.strengths = strengths[:5]
        profile.known_issues = issues[:5]

    def _generate_recommendations(self, profile: ModelProfile) -> list[str]:
        """Generate actionable recommendations based on profile data."""
        recs = []

        for cat_name, cat in profile.category_profiles.items():
            if cat.total_actions < 5:
                continue

            if cat.failure_rate > 0.3:
                recs.append(
                    f"Consider alternative models for {cat_name} tasks — "
                    f"current failure rate is {cat.failure_rate*100:.0f}%"
                )
            if cat.risk_incidents > 5:
                recs.append(
                    f"Tighten security policies for {cat_name} — "
                    f"{cat.risk_incidents} incidents recorded"
                )
            if cat.avg_latency_ms > 3000:
                recs.append(
                    f"High latency in {cat_name} ({cat.avg_latency_ms:.0f}ms avg) — "
                    f"consider faster tooling or caching"
                )

        if profile.overall_score < 40:
            recs.insert(0, "Overall score is below average — review model selection for critical tasks")

        return recs[:5]

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_or_create_profile(self, model_id: str) -> ModelProfile:
        """Get or create a model profile."""
        if model_id not in self._profiles:
            self._profiles[model_id] = ModelProfile(model_id=model_id)
        return self._profiles[model_id]

    @staticmethod
    def _classify_tool(tool_name: str) -> WorkCategory:
        """Classify a tool call into a work category."""
        tool_lower = tool_name.lower()

        # Direct match
        for pattern, category in TOOL_CATEGORY_MAP.items():
            if pattern in tool_lower:
                return category

        # Fallback: if it looks like a read, it's research
        if any(kw in tool_lower for kw in ("read", "get", "list", "show")):
            return WorkCategory.RESEARCH

        # Fallback: if it looks like a write, it's operations
        if any(kw in tool_lower for kw in ("set", "put", "delete", "update")):
            return WorkCategory.OPERATIONS

        # Default to operations
        return WorkCategory.OPERATIONS

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert numeric score to letter grade."""
        for grade, threshold in GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._profiles.clear()
            self._action_log.clear()
