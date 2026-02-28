"""
Shield Command — Causal Failure Diagnosis Engine (CFDE)

When an agent run fails, CFDE doesn't just show a trace — it diagnoses
WHY it failed using causal inference across all past runs, and suggests
the exact fix. Click to apply.

Integration points:
    - AuditChain: reads execution traces
    - CognitiveMemory: cross-references past episodes
    - ModelRegistry: suggests model swaps based on task-model fitness
    - CIBILEngine: incorporates trust score degradation patterns
"""

import json
import time
import uuid
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import Counter

logger = logging.getLogger("shield.causal_diagnosis")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CausalFactor:
    """A factor that contributed to the failure."""
    variable: str           # "model", "context_tokens", "tool_name", "temperature"
    value: Any
    contribution: float     # 0-1, how much this factor caused the failure
    confidence: float       # 0-1
    evidence: str = ""      # "73% failure rate when context > 4000 tokens"


@dataclass
class FixSuggestion:
    """An actionable suggestion to fix the issue."""
    action: str             # Human-readable action
    fix_type: str           # "swap_model", "reduce_context", "add_step", "change_param"
    params: dict = field(default_factory=dict)
    estimated_improvement: float = 0.0  # 0-1
    auto_applicable: bool = False  # Can be applied with one click


@dataclass
class Diagnosis:
    """Complete root cause analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    run_id: str = ""
    goal: str = ""
    error_type: str = ""
    
    causes: list[CausalFactor] = field(default_factory=list)
    fixes: list[FixSuggestion] = field(default_factory=list)
    
    primary_cause: str = ""
    confidence: float = 0.0
    
    similar_failures: int = 0
    similar_successes: int = 0
    
    diagnosed_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Run History Store
# ---------------------------------------------------------------------------

class RunHistoryStore:
    """Stores and queries agent run history for causal analysis."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS run_history (
        id TEXT PRIMARY KEY,
        goal TEXT,
        agent_id TEXT,
        model_id TEXT,
        status TEXT,
        error_type TEXT DEFAULT '',
        error_message TEXT DEFAULT '',
        context_tokens INTEGER DEFAULT 0,
        tool_calls INTEGER DEFAULT 0,
        duration_ms REAL DEFAULT 0,
        temperature REAL DEFAULT 0,
        task_type TEXT DEFAULT '',
        step_count INTEGER DEFAULT 0,
        created_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_rh_status ON run_history(status);
    CREATE INDEX IF NOT EXISTS idx_rh_model ON run_history(model_id);
    CREATE INDEX IF NOT EXISTS idx_rh_error ON run_history(error_type);
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def record(self, run: dict) -> str:
        """Record a completed run."""
        run_id = run.get("id", str(uuid.uuid4())[:10])
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO run_history 
                   (id, goal, agent_id, model_id, status, error_type, error_message,
                    context_tokens, tool_calls, duration_ms, temperature, task_type, step_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, run.get("goal", ""), run.get("agent_id", ""),
                    run.get("model_id", ""), run.get("status", ""),
                    run.get("error_type", ""), run.get("error_message", ""),
                    run.get("context_tokens", 0), run.get("tool_calls", 0),
                    run.get("duration_ms", 0), run.get("temperature", 0),
                    run.get("task_type", ""), run.get("step_count", 0),
                    time.time(),
                ),
            )
        return run_id

    async def find_similar(self, features: dict, limit: int = 100) -> list[dict]:
        """Find runs with similar characteristics."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM run_history ORDER BY created_at DESC LIMIT ?",
                (limit * 3,),
            ).fetchall()
        
        columns = ["id", "goal", "agent_id", "model_id", "status", "error_type",
                    "error_message", "context_tokens", "tool_calls", "duration_ms",
                    "temperature", "task_type", "step_count", "created_at"]
        
        runs = [dict(zip(columns, row)) for row in rows]
        
        # Score similarity
        scored = []
        for run in runs:
            sim = 0
            if run.get("model_id") == features.get("model"):
                sim += 3
            if run.get("task_type") == features.get("task_type"):
                sim += 2
            if run.get("error_type") == features.get("error_type"):
                sim += 2
            # Context token proximity
            ctx_diff = abs(run.get("context_tokens", 0) - features.get("context_tokens", 0))
            if ctx_diff < 1000:
                sim += 1
            scored.append((sim, run))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    async def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]


# ---------------------------------------------------------------------------
# Causal Diagnosis Engine
# ---------------------------------------------------------------------------

class CausalDiagnosisEngine:
    """
    Diagnoses failure root causes using causal inference across run history.
    
    Usage:
        engine = CausalDiagnosisEngine()
        diagnosis = await engine.diagnose({
            "run_id": "abc123",
            "goal": "Analyze revenue data",
            "model": "gpt-4",
            "context_tokens": 6000,
            "tool_calls": 3,
            "task_type": "analysis",
            "error_type": "timeout",
            "error_message": "Model response exceeded timeout",
        })
        
        print(f"Primary cause: {diagnosis.primary_cause}")
        for fix in diagnosis.fixes:
            print(f"  → {fix.action}")
    """

    def __init__(self, db_path: str = "shield_memory.db", model_registry=None):
        self.store = RunHistoryStore(db_path=db_path)
        self.model_registry = model_registry

    async def diagnose(self, failed_run: dict) -> Diagnosis:
        """
        Diagnose why a run failed.
        
        Args:
            failed_run: dict with keys: run_id, goal, model, context_tokens,
                       tool_calls, task_type, error_type, error_message
        """
        diagnosis = Diagnosis(
            run_id=failed_run.get("run_id", ""),
            goal=failed_run.get("goal", ""),
            error_type=failed_run.get("error_type", ""),
        )

        # Find similar past runs
        similar = await self.store.find_similar(failed_run)
        
        if not similar:
            diagnosis.primary_cause = "Insufficient run history for causal analysis"
            diagnosis.confidence = 0.2
            diagnosis.fixes.append(FixSuggestion(
                action="Run more tasks to build up diagnostic data",
                fix_type="collect_data",
            ))
            return diagnosis

        # Separate successes and failures
        successes = [r for r in similar if r.get("status") in ("completed", "partial")]
        failures = [r for r in similar if r.get("status") == "failed"]
        
        diagnosis.similar_failures = len(failures)
        diagnosis.similar_successes = len(successes)

        # Causal inference: which variables correlate with failure?
        causes = []

        # 1. Model correlation
        model_cause = self._analyze_model_factor(failed_run, successes, failures)
        if model_cause:
            causes.append(model_cause)

        # 2. Context length correlation
        context_cause = self._analyze_context_factor(failed_run, successes, failures)
        if context_cause:
            causes.append(context_cause)

        # 3. Task type correlation
        task_cause = self._analyze_task_factor(failed_run, successes, failures)
        if task_cause:
            causes.append(task_cause)

        # 4. Error pattern analysis
        error_cause = self._analyze_error_pattern(failed_run, failures)
        if error_cause:
            causes.append(error_cause)

        # Sort by contribution
        causes.sort(key=lambda c: c.contribution, reverse=True)
        diagnosis.causes = causes

        if causes:
            diagnosis.primary_cause = causes[0].evidence
            diagnosis.confidence = causes[0].confidence

        # Generate fix suggestions
        diagnosis.fixes = self._generate_fixes(failed_run, causes, successes)

        logger.info(
            f"Diagnosed run {diagnosis.run_id}: primary_cause='{diagnosis.primary_cause}' "
            f"confidence={diagnosis.confidence:.2f} fixes={len(diagnosis.fixes)}"
        )
        return diagnosis

    def _analyze_model_factor(self, failed: dict, successes: list, failures: list) -> Optional[CausalFactor]:
        """Check if the model is a failure factor."""
        model = failed.get("model", "")
        if not model:
            return None

        model_failures = [f for f in failures if f.get("model_id") == model]
        model_successes = [s for s in successes if s.get("model_id") == model]
        total = len(model_failures) + len(model_successes)
        
        if total < 3:
            return None

        failure_rate = len(model_failures) / total
        if failure_rate > 0.4:
            # Find the best performing model for this task type
            best_model = self._find_best_model(failed.get("task_type", ""), successes)
            return CausalFactor(
                variable="model",
                value=model,
                contribution=failure_rate,
                confidence=min(total / 20, 1.0),
                evidence=f"Model '{model}' has {failure_rate*100:.0f}% failure rate for this task type"
                         + (f". Consider switching to '{best_model}'" if best_model else ""),
            )
        return None

    def _analyze_context_factor(self, failed: dict, successes: list, failures: list) -> Optional[CausalFactor]:
        """Check if context length correlates with failure."""
        ctx = failed.get("context_tokens", 0)
        if ctx < 100:
            return None

        # Average context in successes vs failures
        success_ctx = [s.get("context_tokens", 0) for s in successes if s.get("context_tokens", 0) > 0]
        failure_ctx = [f.get("context_tokens", 0) for f in failures if f.get("context_tokens", 0) > 0]

        if not success_ctx or not failure_ctx:
            return None

        avg_success_ctx = sum(success_ctx) / len(success_ctx)
        avg_failure_ctx = sum(failure_ctx) / len(failure_ctx)

        if ctx > avg_success_ctx * 1.5 and avg_failure_ctx > avg_success_ctx:
            return CausalFactor(
                variable="context_tokens",
                value=ctx,
                contribution=0.6,
                confidence=0.7,
                evidence=f"Context length ({ctx} tokens) exceeds successful average ({avg_success_ctx:.0f}). "
                         f"Add a summarization or chunking step to reduce context.",
            )
        return None

    def _analyze_task_factor(self, failed: dict, successes: list, failures: list) -> Optional[CausalFactor]:
        """Check if task type has high failure rate."""
        task_type = failed.get("task_type", "")
        if not task_type:
            return None

        task_failures = [f for f in failures if f.get("task_type") == task_type]
        task_successes = [s for s in successes if s.get("task_type") == task_type]
        total = len(task_failures) + len(task_successes)

        if total < 3:
            return None

        failure_rate = len(task_failures) / total
        if failure_rate > 0.5:
            return CausalFactor(
                variable="task_type",
                value=task_type,
                contribution=failure_rate * 0.5,
                confidence=min(total / 15, 1.0),
                evidence=f"Task type '{task_type}' has {failure_rate*100:.0f}% failure rate across {total} runs",
            )
        return None

    def _analyze_error_pattern(self, failed: dict, failures: list) -> Optional[CausalFactor]:
        """Analyze recurring error patterns."""
        error_type = failed.get("error_type", "")
        if not error_type:
            return None

        same_errors = [f for f in failures if f.get("error_type") == error_type]
        if len(same_errors) >= 3:
            return CausalFactor(
                variable="error_pattern",
                value=error_type,
                contribution=0.5,
                confidence=min(len(same_errors) / 10, 1.0),
                evidence=f"Error '{error_type}' has occurred {len(same_errors)} times. Systematic issue detected.",
            )
        return None

    def _find_best_model(self, task_type: str, successes: list) -> Optional[str]:
        """Find the model with the best success rate for a task type."""
        model_count: Counter = Counter()
        for s in successes:
            if s.get("task_type") == task_type and s.get("model_id"):
                model_count[s["model_id"]] += 1
        return model_count.most_common(1)[0][0] if model_count else None

    def _generate_fixes(self, failed: dict, causes: list, successes: list) -> list[FixSuggestion]:
        """Generate actionable fix suggestions."""
        fixes = []

        for cause in causes:
            if cause.variable == "model":
                best = self._find_best_model(failed.get("task_type", ""), successes)
                if best and best != failed.get("model"):
                    fixes.append(FixSuggestion(
                        action=f"Switch model from '{failed.get('model')}' to '{best}' for this task type",
                        fix_type="swap_model",
                        params={"from": failed.get("model"), "to": best, "task_type": failed.get("task_type")},
                        estimated_improvement=cause.contribution * 0.7,
                        auto_applicable=True,
                    ))

            elif cause.variable == "context_tokens":
                fixes.append(FixSuggestion(
                    action="Add a summarization step to reduce context by ~60% before this node",
                    fix_type="add_step",
                    params={"step_type": "summarize", "target_reduction": 0.6},
                    estimated_improvement=0.5,
                    auto_applicable=True,
                ))

            elif cause.variable == "error_pattern":
                error_type = cause.value
                if "timeout" in str(error_type).lower():
                    fixes.append(FixSuggestion(
                        action="Increase timeout or break task into smaller sub-tasks",
                        fix_type="change_param",
                        params={"param": "timeout_ms", "new_value": 60000},
                        estimated_improvement=0.4,
                    ))
                elif "rate" in str(error_type).lower():
                    fixes.append(FixSuggestion(
                        action="Add rate limiting delay between API calls",
                        fix_type="add_step",
                        params={"step_type": "delay", "delay_ms": 1000},
                        estimated_improvement=0.6,
                        auto_applicable=True,
                    ))

        if not fixes:
            fixes.append(FixSuggestion(
                action="Review the agent configuration and retry with adjusted parameters",
                fix_type="manual_review",
                estimated_improvement=0.3,
            ))

        return fixes

    async def get_stats(self) -> dict:
        return {
            "total_runs_tracked": await self.store.count(),
        }
