"""
Tests for Pipeline Engine — PipelineCompiler, PipelineRunner, PipelineStore, DAG validation.
"""

import asyncio
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.pipeline import (
    PipelineCompiler,
    PipelineRunner,
    PipelineStore,
    PipelineSpec,
    PipelineStep,
    PipelineRunResult,
    PipelineStatus,
    StepType,
    StepStatus,
    _BUILTIN_TEMPLATES,
)


# ---- DAG Models ----

class TestPipelineSpec:
    def test_create_simple(self):
        spec = PipelineSpec(
            name="Test",
            steps=[
                PipelineStep(id="s1", name="Step 1"),
                PipelineStep(id="s2", name="Step 2", depends_on=["s1"]),
            ],
        )
        assert len(spec.steps) == 2
        assert spec.status == PipelineStatus.DRAFT

    def test_get_step(self):
        spec = PipelineSpec(
            name="T",
            steps=[PipelineStep(id="a", name="A"), PipelineStep(id="b", name="B")],
        )
        assert spec.get_step("a").name == "A"
        assert spec.get_step("x") is None

    def test_entry_steps(self):
        spec = PipelineSpec(
            name="T",
            steps=[
                PipelineStep(id="a", name="A"),
                PipelineStep(id="b", name="B", depends_on=["a"]),
                PipelineStep(id="c", name="C"),
            ],
        )
        entries = spec.get_entry_steps()
        assert len(entries) == 2
        assert {s.id for s in entries} == {"a", "c"}

    def test_get_next_steps(self):
        spec = PipelineSpec(
            name="T",
            steps=[
                PipelineStep(id="a", name="A", on_success=["b"]),
                PipelineStep(id="b", name="B", depends_on=["a"]),
            ],
        )
        nexts = spec.get_next_steps("a", success=True)
        assert len(nexts) == 1
        assert nexts[0].id == "b"

    def test_validate_valid_dag(self):
        spec = PipelineSpec(
            name="T",
            steps=[
                PipelineStep(id="a", name="A", on_success=["b"]),
                PipelineStep(id="b", name="B", depends_on=["a"], on_success=["c"]),
                PipelineStep(id="c", name="C", depends_on=["b"]),
            ],
        )
        valid, msg = spec.validate_dag()
        assert valid is True

    def test_validate_invalid_ref(self):
        spec = PipelineSpec(
            name="T",
            steps=[
                PipelineStep(id="a", name="A", depends_on=["nonexistent"]),
            ],
        )
        valid, msg = spec.validate_dag()
        assert valid is False
        assert "nonexistent" in msg

    def test_validate_empty(self):
        spec = PipelineSpec(name="Empty")
        valid, msg = spec.validate_dag()
        assert valid is True


class TestPipelineStep:
    def test_defaults(self):
        step = PipelineStep(id="t", name="Test")
        assert step.type == StepType.TOOL_CALL
        assert step.status == StepStatus.PENDING
        assert step.depends_on == []

    def test_llm_step(self):
        step = PipelineStep(
            id="llm", name="AI Step", type=StepType.LLM_CALL,
            model_task_category="code_generation",
            prompt_template="Generate code for {{task}}",
        )
        assert step.type == StepType.LLM_CALL
        assert "{{task}}" in step.prompt_template


# ---- Pipeline Compiler ----

class TestPipelineCompiler:
    def test_template_matching(self):
        compiler = PipelineCompiler()
        spec = asyncio.get_event_loop().run_until_complete(
            compiler.compile("Monitor GitHub PRs and notify on Slack")
        )
        assert spec is not None
        assert len(spec.steps) > 0
        assert "slack" in spec.name.lower() or "github" in spec.name.lower()

    def test_email_template(self):
        compiler = PipelineCompiler()
        spec = asyncio.get_event_loop().run_until_complete(
            compiler.compile("Process customer support emails and create Jira tickets")
        )
        assert len(spec.steps) > 0

    def test_data_analysis_template(self):
        compiler = PipelineCompiler()
        spec = asyncio.get_event_loop().run_until_complete(
            compiler.compile("Query database and generate analytics report")
        )
        assert len(spec.steps) > 0

    def test_deploy_template(self):
        compiler = PipelineCompiler()
        spec = asyncio.get_event_loop().run_until_complete(
            compiler.compile("Deploy application and monitor health")
        )
        assert len(spec.steps) > 0

    def test_no_match_creates_custom(self):
        compiler = PipelineCompiler()
        spec = asyncio.get_event_loop().run_until_complete(
            compiler.compile("Do something completely unique and unmatched XYZ123")
        )
        assert spec is not None
        assert spec.name == "Custom Pipeline"
        assert len(spec.steps) == 1

    def test_list_templates(self):
        compiler = PipelineCompiler()
        templates = compiler.list_templates()
        assert len(templates) == 5
        assert all("name" in t for t in templates)

    def test_json_extraction(self):
        compiler = PipelineCompiler()
        # Direct JSON
        assert compiler._extract_json('{"key": "val"}') == {"key": "val"}
        # Code block
        assert compiler._extract_json('```json\n{"key": "val"}\n```') == {"key": "val"}
        # Embedded in text
        assert compiler._extract_json('Here is: {"key": "val"} done') == {"key": "val"}
        # Invalid
        assert compiler._extract_json('no json here') is None


# ---- Pipeline Runner ----

class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_run_simple(self):
        runner = PipelineRunner()
        pipeline = PipelineSpec(
            name="Test",
            steps=[
                PipelineStep(
                    id="s1", name="Delay", type=StepType.DELAY,
                    delay_seconds=0.01,
                ),
            ],
        )
        result = await runner.run(pipeline)
        assert result.status == PipelineStatus.COMPLETED
        assert result.steps_completed == 1
        assert result.steps_failed == 0
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_run_multi_step(self):
        runner = PipelineRunner()
        pipeline = PipelineSpec(
            name="Multi",
            steps=[
                PipelineStep(
                    id="s1", name="Step 1", type=StepType.DELAY,
                    delay_seconds=0.01,
                ),
                PipelineStep(
                    id="s2", name="Step 2", type=StepType.DELAY,
                    delay_seconds=0.01, depends_on=["s1"],
                ),
                PipelineStep(
                    id="s3", name="Step 3", type=StepType.DELAY,
                    delay_seconds=0.01, depends_on=["s2"],
                ),
            ],
        )
        result = await runner.run(pipeline)
        assert result.status == PipelineStatus.COMPLETED
        assert result.steps_completed == 3

    @pytest.mark.asyncio
    async def test_run_with_context(self):
        runner = PipelineRunner()
        pipeline = PipelineSpec(
            name="Context Test",
            steps=[
                PipelineStep(
                    id="s1", name="Transform", type=StepType.TRANSFORM,
                    parameters={"type": "extract", "key": "input_data"},
                ),
            ],
        )
        result = await runner.run(
            pipeline, initial_context={"input_data": {"value": 42}}
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_template_resolution(self):
        runner = PipelineRunner()
        resolved = runner._resolve_template(
            "Hello {{name}}, your score is {{score}}",
            {"name": "Shield", "score": "100"},
        )
        assert resolved == "Hello Shield, your score is 100"

    @pytest.mark.asyncio
    async def test_nested_template(self):
        runner = PipelineRunner()
        resolved = runner._resolve_template(
            "Result: {{step_1.content}}",
            {"step_1": {"content": "test output"}},
        )
        assert resolved == "Result: test output"

    @pytest.mark.asyncio
    async def test_tool_call_without_gateway(self):
        runner = PipelineRunner()
        pipeline = PipelineSpec(
            name="No Gateway",
            steps=[
                PipelineStep(
                    id="t1", name="Tool", type=StepType.TOOL_CALL,
                    tool_name="test_tool", parameters={"key": "val"},
                ),
            ],
        )
        result = await runner.run(pipeline)
        assert result.status == PipelineStatus.COMPLETED
        # Should complete with simulated result

    @pytest.mark.asyncio
    async def test_step_callback(self):
        called = []
        def on_step(step, pipeline):
            called.append(step.id)

        runner = PipelineRunner(on_step_complete=on_step)
        pipeline = PipelineSpec(
            name="CB",
            steps=[
                PipelineStep(id="a", name="A", type=StepType.DELAY, delay_seconds=0.01),
                PipelineStep(id="b", name="B", type=StepType.DELAY, delay_seconds=0.01, depends_on=["a"]),
            ],
        )
        await runner.run(pipeline)
        assert called == ["a", "b"]


# ---- Pipeline Store ----

class TestPipelineStore:
    def test_save_and_load(self, tmp_path):
        store = PipelineStore(pipelines_dir=str(tmp_path))
        spec = PipelineSpec(
            name="Saved Pipeline",
            steps=[PipelineStep(id="s1", name="Step 1")],
        )
        store.save(spec)

        loaded = store.load(spec.id)
        assert loaded is not None
        assert loaded.name == "Saved Pipeline"
        assert len(loaded.steps) == 1

    def test_list_all(self, tmp_path):
        store = PipelineStore(pipelines_dir=str(tmp_path))
        store.save(PipelineSpec(name="A", steps=[]))
        store.save(PipelineSpec(name="B", steps=[]))
        items = store.list_all()
        assert len(items) == 2

    def test_delete(self, tmp_path):
        store = PipelineStore(pipelines_dir=str(tmp_path))
        spec = PipelineSpec(name="Del", steps=[])
        store.save(spec)
        assert store.delete(spec.id) is True
        assert store.get(spec.id) is None

    def test_delete_nonexistent(self, tmp_path):
        store = PipelineStore(pipelines_dir=str(tmp_path))
        assert store.delete("xxx") is False


# ---- Built-in Templates ----

class TestBuiltinTemplates:
    def test_templates_exist(self):
        assert len(_BUILTIN_TEMPLATES) == 5

    def test_templates_valid(self):
        for t in _BUILTIN_TEMPLATES:
            valid, msg = t.validate_dag()
            assert valid, f"Template '{t.name}' is invalid: {msg}"

    def test_template_names(self):
        names = {t.name for t in _BUILTIN_TEMPLATES}
        assert "GitHub to Slack Notifier" in names
        assert "Deploy and Monitor" in names
        assert "Data Analysis Report" in names


# ---- Imports ----

class TestImports:
    def test_from_package(self):
        from agentvault import PipelineCompiler, PipelineRunner, PipelineStore
        assert PipelineCompiler is not None

    def test_in_all(self):
        import agentvault
        assert "PipelineCompiler" in agentvault.__all__
        assert "PipelineRunner" in agentvault.__all__
        assert "PipelineSpec" in agentvault.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
