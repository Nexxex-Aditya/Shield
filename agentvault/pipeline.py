"""
Shield Command — Pipeline Engine

The core product layer that transforms Shield from middleware into a platform.

Pipelines are directed acyclic graphs (DAGs) of steps. Each step represents
a tool call or transformation that runs through Shield's 12-step security
pipeline. Pipelines can be:
    - Designed from natural language via LLM (PipelineCompiler)
    - Loaded from YAML/JSON files
    - Executed step by step (PipelineRunner)
    - Monitored in real-time

Architecture:
    Description → PipelineCompiler → PipelineSpec (DAG) → PipelineRunner → Results

Key classes:
    PipelineStep    — Single node in the DAG
    PipelineSpec    — Full pipeline definition (steps + edges + metadata)
    PipelineCompiler — LLM-powered: description → PipelineSpec
    PipelineRunner  — Executes a PipelineSpec through MCPGateway
    PipelineStore   — CRUD for pipeline YAML/JSON files
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
import yaml
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("shield.pipeline")


# ---------------------------------------------------------------------------
# Pipeline Data Models
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    """Types of pipeline steps."""
    TOOL_CALL = "tool_call"          # Call a tool through MCPGateway
    TRANSFORM = "transform"          # Transform/filter data between steps
    CONDITION = "condition"          # Branching logic
    PARALLEL = "parallel"            # Run multiple steps in parallel
    HUMAN_REVIEW = "human_review"    # Pause for human approval
    LLM_CALL = "llm_call"           # Call an LLM directly (via ModelRegistry)
    WEBHOOK = "webhook"              # Send/receive HTTP webhook
    DELAY = "delay"                  # Wait for a duration


class StepStatus(str, Enum):
    """Execution status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"              # Waiting for human review or condition


class PipelineStatus(str, Enum):
    """Overall pipeline execution status."""
    DRAFT = "draft"                  # Not yet executed
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"                # Waiting for human intervention
    CANCELLED = "cancelled"


class PipelineStep(BaseModel):
    """A single node in the pipeline DAG."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                             # Human-readable step name
    type: StepType = StepType.TOOL_CALL
    description: str = ""

    # Tool call config
    tool_name: str = ""                   # Which tool/connector to use
    parameters: dict[str, Any] = Field(default_factory=dict)

    # LLM call config
    model_task_category: str = "general"  # Route to best model for this category
    prompt_template: str = ""             # Prompt with {{variable}} placeholders

    # Webhook config
    url: str = ""
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)

    # Condition config
    condition: str = ""                   # Expression to evaluate

    # Delay config
    delay_seconds: float = 0.0

    # DAG connections
    depends_on: list[str] = Field(default_factory=list)   # step IDs this depends on
    on_success: list[str] = Field(default_factory=list)    # next steps on success
    on_failure: list[str] = Field(default_factory=list)    # next steps on failure

    # Execution state (filled at runtime)
    status: StepStatus = StepStatus.PENDING
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0

    # Security (passed through Shield pipeline)
    agent_id: str = "pipeline-agent"
    session_id: str = ""
    security_decision: Optional[str] = None
    audit_event_id: Optional[str] = None


class PipelineSpec(BaseModel):
    """Full pipeline definition — the DAG."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str
    description: str = ""
    version: str = "1.0"
    author: str = "shield"

    # The DAG
    steps: list[PipelineStep] = Field(default_factory=list)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    trigger: str = "manual"             # manual, schedule, webhook, event
    schedule: str = ""                  # cron expression if trigger=schedule
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Execution state
    status: PipelineStatus = PipelineStatus.DRAFT
    run_count: int = 0
    last_run: Optional[str] = None
    last_duration_ms: float = 0.0

    def get_step(self, step_id: str) -> Optional[PipelineStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def get_entry_steps(self) -> list[PipelineStep]:
        """Get steps with no dependencies (DAG entry points)."""
        return [s for s in self.steps if not s.depends_on]

    def get_next_steps(self, step_id: str, success: bool = True) -> list[PipelineStep]:
        """Get steps to run after a given step completes."""
        step = self.get_step(step_id)
        if not step:
            return []
        next_ids = step.on_success if success else step.on_failure
        # Also get steps that depend_on this step
        dependent = [s for s in self.steps if step_id in s.depends_on]
        # Combine, deduplicate
        all_ids = set(next_ids) | {s.id for s in dependent}
        return [s for s in self.steps if s.id in all_ids]

    def validate_dag(self) -> tuple[bool, str]:
        """Validate the DAG: no cycles, all references valid."""
        step_ids = {s.id for s in self.steps}

        # Check all references are valid
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    return False, f"Step '{step.id}' depends on unknown step '{dep}'"
            for nxt in step.on_success + step.on_failure:
                if nxt not in step_ids:
                    return False, f"Step '{step.id}' references unknown step '{nxt}'"

        # Check for cycles via topological sort
        visited: set[str] = set()
        in_stack: set[str] = set()

        def has_cycle(sid: str) -> bool:
            if sid in in_stack:
                return True
            if sid in visited:
                return False
            visited.add(sid)
            in_stack.add(sid)
            step = self.get_step(sid)
            if step:
                for nxt in step.on_success + step.on_failure:
                    if has_cycle(nxt):
                        return True
                for dep_step in self.steps:
                    if sid in dep_step.depends_on:
                        if has_cycle(dep_step.id):
                            return True
            in_stack.discard(sid)
            return False

        for s in self.steps:
            if has_cycle(s.id):
                return False, f"Cycle detected involving step '{s.id}'"

        return True, "Valid"


class PipelineRunResult(BaseModel):
    """Result of executing a pipeline."""
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    total_steps: int = 0
    duration_ms: float = 0.0
    step_results: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline Compiler (Description → DAG)
# ---------------------------------------------------------------------------

# System prompt for the LLM to generate pipeline DAGs
_COMPILER_SYSTEM_PROMPT = """You are Shield Pipeline Compiler. Your job is to convert a natural language 
process description into a structured pipeline DAG (directed acyclic graph).

Output ONLY valid JSON matching this schema:
{
  "name": "pipeline name",
  "description": "what this pipeline does",
  "steps": [
    {
      "id": "step_1",
      "name": "Human-readable step name",
      "type": "tool_call|transform|condition|llm_call|webhook|delay|human_review",
      "description": "What this step does",
      "tool_name": "tool name if type=tool_call",
      "parameters": {},
      "model_task_category": "general|code_generation|data_analysis|conversation|fast",
      "prompt_template": "prompt if type=llm_call, use {{variable}} for placeholders",
      "url": "URL if type=webhook",
      "condition": "expression if type=condition",
      "delay_seconds": 0,
      "depends_on": ["step IDs this step waits for"],
      "on_success": ["step IDs to run next on success"],
      "on_failure": ["step IDs to run on failure"]
    }
  ],
  "tags": ["relevant", "tags"]
}

Rules:
1. Every step needs a unique short id (e.g. "fetch_data", "notify_team")
2. Use depends_on to express ordering — step B depends on step A
3. Use tool_call for actions (API calls, database queries, file ops)
4. Use llm_call for AI tasks (summarize, analyze, classify, generate)
5. Use condition for if/else branching
6. Use webhook for HTTP integrations
7. Include error handling: set on_failure steps where appropriate
8. Keep step names concise and descriptive
9. Output ONLY the JSON, no markdown, no explanation"""


class PipelineCompiler:
    """
    Converts natural language descriptions into executable pipeline DAGs.
    
    Uses ModelRegistry to route the compilation to the best available model.
    Falls back to template-based generation if no LLM is available.
    """

    def __init__(self, model_registry=None) -> None:
        self._model_registry = model_registry
        self._templates: dict[str, PipelineSpec] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load built-in pipeline templates."""
        self._templates = {
            t.name.lower(): t for t in _BUILTIN_TEMPLATES
        }

    async def compile(
        self,
        description: str,
        context: Optional[dict] = None,
    ) -> PipelineSpec:
        """
        Compile a natural language description into a PipelineSpec.
        
        Args:
            description: What the user wants ("Monitor GitHub PRs and notify on Slack")
            context: Optional context (available tools, connectors, etc.)
            
        Returns:
            PipelineSpec ready for execution
        """
        # Try LLM-based compilation first
        if self._model_registry:
            adapter = self._model_registry.get_adapter_for_task("pipeline_design")
            if adapter:
                return await self._compile_with_llm(description, adapter, context)

        # Fallback: template matching
        return self._compile_from_templates(description)

    async def _compile_with_llm(
        self,
        description: str,
        adapter,
        context: Optional[dict] = None,
    ) -> PipelineSpec:
        """Use LLM to generate a pipeline from description."""
        user_prompt = f"Create a pipeline for: {description}"
        if context:
            user_prompt += f"\n\nAvailable tools: {json.dumps(context.get('tools', []))}"
            user_prompt += f"\nAvailable connectors: {json.dumps(context.get('connectors', []))}"

        try:
            result = await adapter.generate(
                messages=[
                    {"role": "system", "content": _COMPILER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Low temp for structured output
                max_tokens=4096,
            )

            content = result.get("content", "")

            # Parse the JSON from the LLM response
            pipeline_data = self._extract_json(content)
            if not pipeline_data:
                logger.warning("LLM returned non-JSON, falling back to templates")
                return self._compile_from_templates(description)

            # Build PipelineSpec from LLM output
            steps = []
            for step_data in pipeline_data.get("steps", []):
                step = PipelineStep(
                    id=step_data.get("id", str(uuid.uuid4())[:8]),
                    name=step_data.get("name", "Untitled Step"),
                    type=StepType(step_data.get("type", "tool_call")),
                    description=step_data.get("description", ""),
                    tool_name=step_data.get("tool_name", ""),
                    parameters=step_data.get("parameters", {}),
                    model_task_category=step_data.get("model_task_category", "general"),
                    prompt_template=step_data.get("prompt_template", ""),
                    url=step_data.get("url", ""),
                    method=step_data.get("method", "POST"),
                    condition=step_data.get("condition", ""),
                    delay_seconds=step_data.get("delay_seconds", 0.0),
                    depends_on=step_data.get("depends_on", []),
                    on_success=step_data.get("on_success", []),
                    on_failure=step_data.get("on_failure", []),
                )
                steps.append(step)

            spec = PipelineSpec(
                name=pipeline_data.get("name", "Generated Pipeline"),
                description=pipeline_data.get("description", description),
                steps=steps,
                tags=pipeline_data.get("tags", []),
                author="llm-compiler",
            )

            # Validate
            valid, msg = spec.validate_dag()
            if not valid:
                logger.warning("LLM-generated DAG invalid: %s. Falling back.", msg)
                return self._compile_from_templates(description)

            logger.info("Compiled pipeline '%s' with %d steps via LLM", spec.name, len(steps))
            return spec

        except Exception as e:
            logger.error("LLM compilation failed: %s. Falling back to templates.", e)
            return self._compile_from_templates(description)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM output (handles markdown code blocks)."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` blocks
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i, ch in enumerate(text[brace_start:], brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    def _compile_from_templates(self, description: str) -> PipelineSpec:
        """Match description to a built-in template."""
        desc_lower = description.lower()

        # Simple keyword-based matching
        best_match = None
        best_score = 0

        for name, template in self._templates.items():
            score = 0
            keywords = name.split("_") + template.tags
            for kw in keywords:
                if kw.lower() in desc_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = template

        if best_match and best_score > 0:
            # Clone the template with a new ID
            spec = best_match.model_copy(deep=True)
            spec.id = str(uuid.uuid4())[:12]
            spec.description = description
            spec.created_at = datetime.utcnow().isoformat()
            logger.info("Matched template '%s' (score=%d)", spec.name, best_score)
            return spec

        # No match — create a minimal single-step pipeline
        logger.info("No template match; creating minimal pipeline")
        return PipelineSpec(
            name="Custom Pipeline",
            description=description,
            steps=[
                PipelineStep(
                    id="step_1",
                    name="Process Request",
                    type=StepType.LLM_CALL,
                    description=description,
                    model_task_category="general",
                    prompt_template=description,
                ),
            ],
            tags=["custom", "auto-generated"],
        )

    def list_templates(self) -> list[dict]:
        """List available pipeline templates."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "steps": len(t.steps),
                "tags": t.tags,
            }
            for t in self._templates.values()
        ]


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Executes a PipelineSpec step by step through Shield's security pipeline.
    
    Each tool_call step goes through MCPGateway's 12-step security pipeline.
    LLM calls go through ModelRegistry.
    Results flow between steps via a shared context dict.
    """

    def __init__(
        self,
        gateway=None,
        model_registry=None,
        connector_executor=None,
        on_step_complete: Optional[Callable] = None,
    ) -> None:
        self._gateway = gateway
        self._model_registry = model_registry
        self._connector_executor = connector_executor
        self._on_step_complete = on_step_complete  # callback for real-time updates

    async def run(
        self,
        pipeline: PipelineSpec,
        initial_context: Optional[dict] = None,
    ) -> PipelineRunResult:
        """
        Execute a pipeline from start to finish.
        
        Args:
            pipeline: The PipelineSpec to execute
            initial_context: Initial data available to all steps
            
        Returns:
            PipelineRunResult with status and per-step results
        """
        start_time = time.monotonic()
        context = dict(initial_context or {})
        pipeline.status = PipelineStatus.RUNNING
        pipeline.run_count += 1

        session_id = f"pipeline-{pipeline.id}-{pipeline.run_count}"

        result = PipelineRunResult(
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            status=PipelineStatus.RUNNING,
            total_steps=len(pipeline.steps),
        )

        # Assign session ID to all steps
        for step in pipeline.steps:
            step.session_id = session_id
            step.status = StepStatus.PENDING

        try:
            # Execute steps in topological order
            completed_steps: set[str] = set()
            failed_steps: set[str] = set()

            while True:
                # Find ready steps (all dependencies satisfied)
                ready = [
                    s for s in pipeline.steps
                    if s.status == StepStatus.PENDING
                    and all(d in completed_steps for d in s.depends_on)
                    and not any(d in failed_steps for d in s.depends_on)
                ]

                if not ready:
                    # Check if we're done or stuck
                    pending = [s for s in pipeline.steps if s.status == StepStatus.PENDING]
                    if not pending:
                        break  # All done
                    # Some steps are pending but have failed dependencies — skip them
                    for s in pending:
                        if any(d in failed_steps for d in s.depends_on):
                            s.status = StepStatus.SKIPPED
                            result.steps_skipped += 1
                    if not [s for s in pipeline.steps if s.status == StepStatus.PENDING]:
                        break  # All resolved
                    break  # Stuck (shouldn't happen with a valid DAG)

                # Execute ready steps (could be parallel in the future)
                for step in ready:
                    step_result = await self._execute_step(step, context, session_id)

                    if step.status == StepStatus.COMPLETED:
                        completed_steps.add(step.id)
                        result.steps_completed += 1
                        # Store result in context for downstream steps
                        context[f"step_{step.id}"] = step_result
                        context["last_result"] = step_result
                    elif step.status == StepStatus.FAILED:
                        failed_steps.add(step.id)
                        result.steps_failed += 1
                        # Execute failure handlers
                        for fail_id in step.on_failure:
                            fail_step = pipeline.get_step(fail_id)
                            if fail_step and fail_step.status == StepStatus.PENDING:
                                await self._execute_step(fail_step, context, session_id)

                    result.step_results.append({
                        "step_id": step.id,
                        "step_name": step.name,
                        "status": step.status.value,
                        "result": step.result,
                        "error": step.error,
                        "duration_ms": step.duration_ms,
                        "security_decision": step.security_decision,
                    })

                    # Notify callback
                    if self._on_step_complete:
                        try:
                            self._on_step_complete(step, pipeline)
                        except Exception:
                            pass

            # Determine final status
            if result.steps_failed > 0:
                result.status = PipelineStatus.FAILED
                pipeline.status = PipelineStatus.FAILED
            else:
                result.status = PipelineStatus.COMPLETED
                pipeline.status = PipelineStatus.COMPLETED

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            pipeline.status = PipelineStatus.FAILED
            logger.error("Pipeline '%s' failed: %s", pipeline.name, e)

        # Finalize timing
        duration = (time.monotonic() - start_time) * 1000
        result.duration_ms = round(duration, 1)
        result.completed_at = datetime.utcnow().isoformat()
        pipeline.last_run = result.completed_at
        pipeline.last_duration_ms = result.duration_ms

        logger.info(
            "Pipeline '%s' %s: %d/%d steps in %.0fms",
            pipeline.name, result.status.value,
            result.steps_completed, result.total_steps, result.duration_ms,
        )

        return result

    async def _execute_step(
        self,
        step: PipelineStep,
        context: dict,
        session_id: str,
    ) -> Optional[dict]:
        """Execute a single pipeline step."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow().isoformat()
        start = time.monotonic()

        try:
            result = None

            if step.type == StepType.TOOL_CALL:
                result = await self._exec_tool_call(step, context, session_id)

            elif step.type == StepType.LLM_CALL:
                result = await self._exec_llm_call(step, context)

            elif step.type == StepType.TRANSFORM:
                result = await self._exec_transform(step, context)

            elif step.type == StepType.CONDITION:
                result = await self._exec_condition(step, context)

            elif step.type == StepType.WEBHOOK:
                result = await self._exec_webhook(step, context)

            elif step.type == StepType.DELAY:
                import asyncio
                await asyncio.sleep(step.delay_seconds)
                result = {"delayed": step.delay_seconds}

            elif step.type == StepType.HUMAN_REVIEW:
                step.status = StepStatus.WAITING
                result = {"waiting_for": "human_review"}
                return result

            step.status = StepStatus.COMPLETED
            step.result = result
            step.duration_ms = round((time.monotonic() - start) * 1000, 1)
            step.completed_at = datetime.utcnow().isoformat()
            return result

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.duration_ms = round((time.monotonic() - start) * 1000, 1)
            step.completed_at = datetime.utcnow().isoformat()
            logger.error("Step '%s' failed: %s", step.name, e)
            return None

    async def _exec_tool_call(
        self, step: PipelineStep, context: dict, session_id: str,
    ) -> dict:
        """Execute a tool call through MCPGateway or ConnectorExecutor."""
        # Resolve parameters with context variables
        params = self._resolve_params(step.parameters, context)

        # Try MCPGateway first (full security pipeline)
        if self._gateway:
            result = await self._gateway.handle_tool_call(
                agent_id=step.agent_id,
                session_id=session_id,
                tool_name=step.tool_name,
                parameters=params,
            )
            step.security_decision = result.get("decision", "unknown")
            step.audit_event_id = result.get("audit_event_id")
            return result

        # Fallback: ConnectorExecutor (direct connector routing)
        if self._connector_executor:
            return await self._connector_executor.route(step.tool_name, params)

        return {"error": "No gateway or connector executor configured", "simulated": True}

    async def _exec_llm_call(
        self, step: PipelineStep, context: dict,
    ) -> dict:
        """Execute an LLM call via ModelRegistry."""
        if not self._model_registry:
            return {"error": "No model registry configured", "simulated": True}

        adapter = self._model_registry.get_adapter_for_task(
            step.model_task_category
        )
        if not adapter:
            return {"error": f"No model available for '{step.model_task_category}'"}

        # Resolve prompt template with context
        prompt = self._resolve_template(step.prompt_template, context)

        result = await adapter.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )

        return {
            "content": result.get("content", ""),
            "model": result.get("model", "unknown"),
            "tokens": result.get("usage", {}),
        }

    async def _exec_transform(
        self, step: PipelineStep, context: dict,
    ) -> dict:
        """Execute a data transformation step."""
        # Simple transform: extract/filter/map data from context
        transform_type = step.parameters.get("type", "passthrough")

        if transform_type == "extract":
            key = step.parameters.get("key", "last_result")
            field = step.parameters.get("field", "")
            data = context.get(key, {})
            if field and isinstance(data, dict):
                return {"value": data.get(field)}
            return {"value": data}

        elif transform_type == "filter":
            key = step.parameters.get("key", "last_result")
            condition_field = step.parameters.get("field", "")
            condition_value = step.parameters.get("value", "")
            data = context.get(key, {})
            if isinstance(data, list):
                filtered = [
                    item for item in data
                    if str(item.get(condition_field, "")) == str(condition_value)
                ]
                return {"filtered": filtered, "count": len(filtered)}
            return {"value": data}

        # Default: pass through last result
        return {"value": context.get("last_result", {})}

    async def _exec_condition(
        self, step: PipelineStep, context: dict,
    ) -> dict:
        """Evaluate a condition and determine next steps."""
        condition = step.condition
        if not condition:
            return {"result": True}

        # Simple expression evaluation against context
        try:
            # Replace context references: {{key}} → context value
            resolved = self._resolve_template(condition, context)
            # Basic eval (safe: only string comparisons)
            result = bool(resolved.strip())
            return {"result": result, "condition": condition}
        except Exception as e:
            return {"result": False, "error": str(e)}

    async def _exec_webhook(
        self, step: PipelineStep, context: dict,
    ) -> dict:
        """Execute an HTTP webhook call."""
        import httpx

        url = self._resolve_template(step.url, context)
        body = self._resolve_params(step.parameters, context)
        headers = dict(step.headers)

        async with httpx.AsyncClient(timeout=30.0) as client:
            if step.method.upper() == "GET":
                resp = await client.get(url, headers=headers, params=body)
            else:
                resp = await client.request(
                    step.method.upper(), url, headers=headers, json=body,
                )
            return {
                "status_code": resp.status_code,
                "body": resp.text[:2000],  # Limit response size
                "headers": dict(resp.headers),
            }

    def _resolve_params(self, params: dict, context: dict) -> dict:
        """Replace {{variable}} placeholders in parameter values."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_template(value, context)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_params(value, context)
            else:
                resolved[key] = value
        return resolved

    def _resolve_template(self, template: str, context: dict) -> str:
        """Replace {{variable}} placeholders with context values."""
        import re
        def replacer(match):
            key = match.group(1).strip()
            # Support dot notation: {{step_1.content}}
            parts = key.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, "")
                else:
                    value = ""
                    break
            return str(value)
        return re.sub(r"\{\{(.+?)\}\}", replacer, template)


# ---------------------------------------------------------------------------
# Pipeline Store (YAML/JSON persistence)
# ---------------------------------------------------------------------------

class PipelineStore:
    """
    CRUD for pipeline definitions stored as YAML/JSON files.
    """

    def __init__(self, pipelines_dir: str = "pipelines") -> None:
        self._dir = Path(pipelines_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pipelines: dict[str, PipelineSpec] = {}

    def save(self, pipeline: PipelineSpec) -> str:
        """Save a pipeline to YAML file."""
        filepath = self._dir / f"{pipeline.id}.yaml"
        data = pipeline.model_dump(mode="json")
        filepath.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        self._pipelines[pipeline.id] = pipeline
        return str(filepath)

    def load(self, pipeline_id: str) -> Optional[PipelineSpec]:
        """Load a pipeline from YAML file."""
        filepath = self._dir / f"{pipeline_id}.yaml"
        if not filepath.exists():
            return None
        try:
            data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
            spec = PipelineSpec(**data)
            self._pipelines[spec.id] = spec
            return spec
        except Exception as e:
            logger.error("Failed to load pipeline %s: %s", pipeline_id, e)
            return None

    def load_all(self) -> list[PipelineSpec]:
        """Load all pipelines from the directory."""
        pipelines = []
        for fp in self._dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(fp.read_text(encoding="utf-8"))
                spec = PipelineSpec(**data)
                self._pipelines[spec.id] = spec
                pipelines.append(spec)
            except Exception as e:
                logger.error("Failed to load %s: %s", fp.name, e)
        return pipelines

    def delete(self, pipeline_id: str) -> bool:
        """Delete a pipeline file."""
        filepath = self._dir / f"{pipeline_id}.yaml"
        if filepath.exists():
            filepath.unlink()
            self._pipelines.pop(pipeline_id, None)
            return True
        return False

    def list_all(self) -> list[dict]:
        """List all saved pipelines (metadata only)."""
        if not self._pipelines:
            self.load_all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "steps": len(p.steps),
                "status": p.status.value,
                "tags": p.tags,
                "run_count": p.run_count,
                "last_run": p.last_run,
                "created_at": p.created_at,
            }
            for p in self._pipelines.values()
        ]

    def get(self, pipeline_id: str) -> Optional[PipelineSpec]:
        """Get a pipeline by ID (from cache or disk)."""
        if pipeline_id in self._pipelines:
            return self._pipelines[pipeline_id]
        return self.load(pipeline_id)


# ---------------------------------------------------------------------------
# Built-in Pipeline Templates
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: list[PipelineSpec] = [
    PipelineSpec(
        id="tmpl-github-slack",
        name="GitHub to Slack Notifier",
        description="Monitor GitHub events and notify team on Slack",
        tags=["github", "slack", "notification", "monitoring"],
        steps=[
            PipelineStep(
                id="fetch_events",
                name="Fetch GitHub Events",
                type=StepType.TOOL_CALL,
                tool_name="github_list_events",
                parameters={"repo": "{{repo}}", "event_type": "push"},
                on_success=["analyze"],
            ),
            PipelineStep(
                id="analyze",
                name="Analyze Events",
                type=StepType.LLM_CALL,
                description="Summarize the events",
                model_task_category="summarization",
                prompt_template="Summarize these GitHub events concisely:\n{{step_fetch_events}}",
                depends_on=["fetch_events"],
                on_success=["notify"],
            ),
            PipelineStep(
                id="notify",
                name="Notify on Slack",
                type=StepType.TOOL_CALL,
                tool_name="slack_send_message",
                parameters={
                    "channel": "{{slack_channel}}",
                    "message": "{{step_analyze.content}}",
                },
                depends_on=["analyze"],
            ),
        ],
    ),
    PipelineSpec(
        id="tmpl-email-jira",
        name="Email to Jira Ticket",
        description="Process incoming emails and create Jira tickets for complaints",
        tags=["email", "jira", "support", "customer", "complaint", "ticket"],
        steps=[
            PipelineStep(
                id="read_email",
                name="Read Incoming Email",
                type=StepType.TOOL_CALL,
                tool_name="email_read_inbox",
                parameters={"filter": "unread", "limit": 10},
                on_success=["classify"],
            ),
            PipelineStep(
                id="classify",
                name="Classify Sentiment",
                type=StepType.LLM_CALL,
                model_task_category="data_analysis",
                prompt_template="Classify the sentiment of this email as POSITIVE, NEUTRAL, or NEGATIVE. Reply with ONLY the classification word.\n\nEmail: {{step_read_email}}",
                depends_on=["read_email"],
                on_success=["check_negative"],
            ),
            PipelineStep(
                id="check_negative",
                name="Check if Negative",
                type=StepType.CONDITION,
                condition="{{step_classify.content}}",
                depends_on=["classify"],
                on_success=["create_ticket"],
            ),
            PipelineStep(
                id="create_ticket",
                name="Create Jira Ticket",
                type=StepType.TOOL_CALL,
                tool_name="jira_create_issue",
                parameters={
                    "project": "{{jira_project}}",
                    "summary": "Customer Complaint: {{step_read_email}}",
                    "type": "Bug",
                },
                depends_on=["check_negative"],
                on_success=["draft_reply"],
            ),
            PipelineStep(
                id="draft_reply",
                name="Draft Response",
                type=StepType.LLM_CALL,
                model_task_category="conversation",
                prompt_template="Draft a professional, empathetic response to this customer complaint:\n{{step_read_email}}\n\nKeep it under 200 words.",
                depends_on=["create_ticket"],
            ),
        ],
    ),
    PipelineSpec(
        id="tmpl-data-report",
        name="Data Analysis Report",
        description="Query a database, analyze results, and generate a report",
        tags=["data", "analysis", "report", "database", "sql", "analytics"],
        steps=[
            PipelineStep(
                id="query_db",
                name="Query Database",
                type=StepType.TOOL_CALL,
                tool_name="postgresql_query",
                parameters={"query": "{{sql_query}}"},
                on_success=["analyze_data"],
            ),
            PipelineStep(
                id="analyze_data",
                name="Analyze Data",
                type=StepType.LLM_CALL,
                model_task_category="data_analysis",
                prompt_template="Analyze this data and provide key insights, trends, and recommendations:\n{{step_query_db}}",
                depends_on=["query_db"],
                on_success=["generate_report"],
            ),
            PipelineStep(
                id="generate_report",
                name="Generate Report",
                type=StepType.LLM_CALL,
                model_task_category="summarization",
                prompt_template="Create a professional report with sections: Executive Summary, Key Findings, Detailed Analysis, Recommendations.\n\nBased on:\n{{step_analyze_data.content}}",
                depends_on=["analyze_data"],
            ),
        ],
    ),
    PipelineSpec(
        id="tmpl-deploy-monitor",
        name="Deploy and Monitor",
        description="Deploy code, run health checks, and alert on failure",
        tags=["deploy", "deployment", "monitor", "health", "devops", "ci", "cd"],
        steps=[
            PipelineStep(
                id="run_tests",
                name="Run Test Suite",
                type=StepType.TOOL_CALL,
                tool_name="shell_execute",
                parameters={"command": "{{test_command}}"},
                on_success=["deploy"],
                on_failure=["notify_failure"],
            ),
            PipelineStep(
                id="deploy",
                name="Deploy Application",
                type=StepType.TOOL_CALL,
                tool_name="shell_execute",
                parameters={"command": "{{deploy_command}}"},
                depends_on=["run_tests"],
                on_success=["health_check"],
                on_failure=["rollback"],
            ),
            PipelineStep(
                id="health_check",
                name="Health Check",
                type=StepType.WEBHOOK,
                url="{{health_url}}",
                method="GET",
                depends_on=["deploy"],
                on_success=["notify_success"],
                on_failure=["rollback"],
            ),
            PipelineStep(
                id="notify_success",
                name="Notify Success",
                type=StepType.TOOL_CALL,
                tool_name="slack_send_message",
                parameters={
                    "channel": "#deployments",
                    "message": "✅ Deployment successful! Health check passed.",
                },
                depends_on=["health_check"],
            ),
            PipelineStep(
                id="rollback",
                name="Rollback Deployment",
                type=StepType.TOOL_CALL,
                tool_name="shell_execute",
                parameters={"command": "{{rollback_command}}"},
                on_success=["notify_failure"],
            ),
            PipelineStep(
                id="notify_failure",
                name="Notify Failure",
                type=StepType.TOOL_CALL,
                tool_name="slack_send_message",
                parameters={
                    "channel": "#deployments",
                    "message": "❌ Deployment failed! {{last_result}}",
                },
            ),
        ],
    ),
    PipelineSpec(
        id="tmpl-content-pipeline",
        name="Content Generation Pipeline",
        description="Research a topic, generate content, review, and publish",
        tags=["content", "blog", "writing", "research", "generate", "publish"],
        steps=[
            PipelineStep(
                id="research",
                name="Research Topic",
                type=StepType.LLM_CALL,
                model_task_category="data_analysis",
                prompt_template="Research the topic '{{topic}}' and provide:\n1. Key facts and statistics\n2. Current trends\n3. Expert opinions\n4. Counterarguments\n\nBe thorough and cite sources where possible.",
                on_success=["draft"],
            ),
            PipelineStep(
                id="draft",
                name="Draft Content",
                type=StepType.LLM_CALL,
                model_task_category="code_generation",
                prompt_template="Write a professional blog post about '{{topic}}' based on this research:\n{{step_research.content}}\n\nFormat: Markdown. Length: 800-1200 words. Tone: {{tone}}.",
                depends_on=["research"],
                on_success=["review"],
            ),
            PipelineStep(
                id="review",
                name="AI Review",
                type=StepType.LLM_CALL,
                model_task_category="general",
                prompt_template="Review this blog post for:\n1. Factual accuracy\n2. Clarity and readability\n3. SEO optimization\n4. Grammar and style\n\nProvide a score (1-10) and specific suggestions.\n\nPost:\n{{step_draft.content}}",
                depends_on=["draft"],
                on_success=["publish"],
            ),
            PipelineStep(
                id="publish",
                name="Publish Content",
                type=StepType.HUMAN_REVIEW,
                description="Human reviews the final draft before publishing",
                depends_on=["review"],
            ),
        ],
    ),
]
