"""
AgentVault — Recommendation Engine

Uses CIBIL score data to make intelligent suggestions about:
    - Best model for a given task category
    - Tools that work well with specific models
    - Known issues and things to watch for
    - Configuration optimizations

Feeds from: CIBILEngine profiles
Serves to: API endpoints + Dashboard

This is where Shield's data network effect becomes user-facing value.
The more customers use Shield, the better the recommendations get.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from .models import Recommendation, WorkCategory
from .cibil import CIBILEngine

logger = logging.getLogger("agentvault.recommendations")

# Minimum actions required before including a model in recommendations
MIN_ACTIONS_FOR_REC = 10
# Minimum confidence to surface a recommendation
MIN_CONFIDENCE = 0.3


class RecommendationEngine:
    """
    Intelligent model/tool/configuration recommendation engine.

    Leverages cross-user CIBIL data to surface actionable insights:
    - "For data analysis, GPT-4o outperforms Claude by 12% success rate"
    - "Model X has a known issue with SQL queries — consider Model Y"
    - "Adding a Redis cache reduced latency by 40% for similar setups"
    """

    def __init__(self, cibil: CIBILEngine) -> None:
        self._cibil = cibil
        self._custom_rules: list[dict[str, Any]] = []

    # ── Model Recommendations ────────────────────────────────────────

    def suggest_model(
        self,
        task_category: str,
        current_model: Optional[str] = None,
        top_n: int = 3,
    ) -> list[Recommendation]:
        """
        Suggest the best models for a specific task category.
        Ranks by CIBIL category score, filtered by minimum data.
        """
        candidates = []

        for model_id, profile in self._cibil._profiles.items():
            cat = profile.category_profiles.get(task_category)
            if not cat or cat.total_actions < MIN_ACTIONS_FOR_REC:
                continue

            candidates.append({
                "model_id": model_id,
                "score": cat.score,
                "success_rate": cat.success_rate,
                "total_actions": cat.total_actions,
                "avg_latency_ms": cat.avg_latency_ms,
                "risk_incidents": cat.risk_incidents,
            })

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        recommendations = []
        for i, c in enumerate(candidates[:top_n]):
            confidence = min(1.0, c["total_actions"] / 100)

            # Build detail text
            detail = (
                f"Score: {c['score']:.0f}/100 | "
                f"Success: {c['success_rate']*100:.0f}% | "
                f"Latency: {c['avg_latency_ms']:.0f}ms | "
                f"Based on {c['total_actions']} actions"
            )

            # Add comparison to current model if provided
            if current_model and current_model != c["model_id"]:
                current = self._cibil.get_profile(current_model)
                if current:
                    curr_cat = current.category_profiles.get(task_category)
                    if curr_cat and curr_cat.total_actions >= MIN_ACTIONS_FOR_REC:
                        delta = c["score"] - curr_cat.score
                        if delta > 0:
                            detail += f" | +{delta:.0f} pts vs your current model"

            rec = Recommendation(
                rec_type="model",
                title=f"{'🥇' if i == 0 else '🥈' if i == 1 else '🥉'} {c['model_id']}",
                detail=detail,
                confidence=round(confidence, 2),
                source_data_points=c["total_actions"],
                model_id=c["model_id"],
                category=task_category,
                tags=["model_suggestion", task_category],
            )
            recommendations.append(rec)

        return recommendations

    def suggest_tools(
        self,
        model_id: str,
        task_category: Optional[str] = None,
        top_n: int = 5,
    ) -> list[Recommendation]:
        """
        Suggest tools that work well with a specific model.
        Based on success rates and frequency of use.
        """
        profile = self._cibil.get_profile(model_id)
        if not profile:
            return []

        tool_scores: dict[str, dict] = {}

        for cat_name, cat in profile.category_profiles.items():
            if task_category and cat_name != task_category:
                continue

            for tool, count in cat.tools_used.items():
                if tool not in tool_scores:
                    tool_scores[tool] = {
                        "count": 0,
                        "category": cat_name,
                        "cat_score": cat.score,
                        "success_rate": cat.success_rate,
                    }
                tool_scores[tool]["count"] += count

        # Sort by usage count
        sorted_tools = sorted(
            tool_scores.items(), key=lambda x: x[1]["count"], reverse=True
        )

        recommendations = []
        for tool, data in sorted_tools[:top_n]:
            confidence = min(1.0, data["count"] / 50)
            rec = Recommendation(
                rec_type="tool",
                title=f"🔧 {tool}",
                detail=(
                    f"Used {data['count']}x in {data['category']} | "
                    f"Category score: {data['cat_score']:.0f} | "
                    f"Success rate: {data['success_rate']*100:.0f}%"
                ),
                confidence=round(confidence, 2),
                source_data_points=data["count"],
                model_id=model_id,
                category=data["category"],
                tags=["tool_suggestion"],
            )
            recommendations.append(rec)

        return recommendations

    def get_warnings(
        self,
        model_id: str,
        task_category: Optional[str] = None,
    ) -> list[Recommendation]:
        """
        Get warnings and known issues for a model.
        Based on failure rates, risk incidents, and performance data.
        """
        profile = self._cibil.get_profile(model_id)
        if not profile:
            return []

        warnings = []

        for cat_name, cat in profile.category_profiles.items():
            if task_category and cat_name != task_category:
                continue
            if cat.total_actions < MIN_ACTIONS_FOR_REC:
                continue

            # High failure rate warning
            if cat.failure_rate > 0.25:
                warnings.append(Recommendation(
                    rec_type="warning",
                    title=f"⚠️ High failure rate in {cat_name}",
                    detail=(
                        f"{cat.failure_rate*100:.0f}% failure rate across "
                        f"{cat.total_actions} actions. Consider an alternative model "
                        f"for {cat_name} tasks."
                    ),
                    confidence=min(1.0, cat.total_actions / 50),
                    source_data_points=cat.total_actions,
                    model_id=model_id,
                    category=cat_name,
                    tags=["warning", "failure_rate"],
                ))

            # Security risk warning
            if cat.risk_incidents > cat.total_actions * 0.1:
                warnings.append(Recommendation(
                    rec_type="warning",
                    title=f"🛡️ Security concerns in {cat_name}",
                    detail=(
                        f"{cat.risk_incidents} security incidents in "
                        f"{cat.total_actions} actions. This model may need "
                        f"tighter policy constraints for {cat_name} work."
                    ),
                    confidence=min(1.0, cat.total_actions / 50),
                    source_data_points=cat.total_actions,
                    model_id=model_id,
                    category=cat_name,
                    tags=["warning", "security"],
                ))

            # Slow performance warning
            if cat.avg_latency_ms > 5000:
                warnings.append(Recommendation(
                    rec_type="warning",
                    title=f"🐌 Slow performance in {cat_name}",
                    detail=(
                        f"Average latency: {cat.avg_latency_ms:.0f}ms. "
                        f"Consider optimizing tool configurations or "
                        f"switching to a faster model for {cat_name}."
                    ),
                    confidence=min(1.0, cat.total_actions / 50),
                    source_data_points=cat.total_actions,
                    model_id=model_id,
                    category=cat_name,
                    tags=["warning", "performance"],
                ))

            # Low score warning
            if cat.score < 30:
                warnings.append(Recommendation(
                    rec_type="warning",
                    title=f"📉 Low score in {cat_name}",
                    detail=(
                        f"CIBIL score {cat.score:.0f}/100 in {cat_name}. "
                        f"This model is not well-suited for {cat_name} tasks "
                        f"based on {cat.total_actions} observed actions."
                    ),
                    confidence=min(1.0, cat.total_actions / 50),
                    source_data_points=cat.total_actions,
                    model_id=model_id,
                    category=cat_name,
                    tags=["warning", "low_score"],
                ))

        return warnings

    # ── Combined Recommendations ─────────────────────────────────────

    def generate_report(
        self,
        model_id: str,
    ) -> dict[str, Any]:
        """
        Generate a full recommendation report for a model.
        Combines report card + tool suggestions + warnings.
        """
        report_card = self._cibil.get_report_card(model_id)
        if not report_card:
            return {"error": f"No data for model '{model_id}'"}

        # Get recommendations for each category
        category_recs = {}
        for cat_name in (report_card.best_categories + report_card.weakest_categories):
            alt_models = self.suggest_model(cat_name, current_model=model_id, top_n=2)
            tools = self.suggest_tools(model_id, task_category=cat_name, top_n=3)
            warns = self.get_warnings(model_id, task_category=cat_name)

            category_recs[cat_name] = {
                "alternative_models": [r.model_dump(mode="json") for r in alt_models],
                "suggested_tools": [r.model_dump(mode="json") for r in tools],
                "warnings": [r.model_dump(mode="json") for r in warns],
            }

        return {
            "report_card": report_card.model_dump(mode="json"),
            "category_insights": category_recs,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_dashboard_summary(self) -> dict[str, Any]:
        """
        Get a summary suitable for the dashboard overview.
        Top models, biggest issues, trending recommendations.
        """
        models = self._cibil.list_models()

        top_models = models[:5] if models else []
        struggling = [m for m in models if m["overall_score"] < 40]

        # Collect all warnings across models
        all_warnings = []
        for m in models[:10]:  # Top 10 models
            warnings = self.get_warnings(m["model_id"])
            for w in warnings:
                all_warnings.append({
                    "model_id": m["model_id"],
                    **w.model_dump(mode="json"),
                })

        return {
            "total_models_tracked": len(models),
            "top_models": top_models,
            "struggling_models": struggling[:5],
            "recent_warnings": all_warnings[:10],
            "categories": [c.value for c in WorkCategory],
        }
