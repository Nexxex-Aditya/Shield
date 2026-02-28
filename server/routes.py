"""
AgentVault — API Routes
REST + WebSocket endpoints for the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("agentvault.routes")

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    agent_id: str
    session_id: str
    tool_name: str
    parameters: dict = {}


class LogRequest(BaseModel):
    agent_id: str
    session_id: str
    trace_id: str
    action_name: str
    tool_name: str
    parameters: dict = {}
    decision: str
    reasoning: str = ""
    result: Optional[dict] = None


class ResolveRequest(BaseModel):
    approved: bool
    resolved_by: str = "admin"


class SimulateRequest(BaseModel):
    candidate_policy_path: str
    audit_events: list[dict] = []


class HoneypotRegisterRequest(BaseModel):
    tool_name: str
    description: str = "Canary tool"
    category: str = "general"


class ConnectRequest(BaseModel):
    connector_id: str
    config: dict = {}
    instance_name: str = ""


class ModelAddRequest(BaseModel):
    name: str
    provider: str                            # openai, anthropic, gemini, ollama
    model_id: str                            # e.g. "gpt-4o", "gemini-2.0-flash"
    api_key: str = ""                        # raw key — encrypted before storage
    base_url: Optional[str] = None
    is_default: bool = False
    task_categories: list[str] = ["general"]
    priority: int = 0
    max_tokens: int = 4096
    temperature: float = 0.7


class ModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    model_id: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_default: Optional[bool] = None
    enabled: Optional[bool] = None
    task_categories: Optional[list[str]] = None
    priority: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class QuickSetupRequest(BaseModel):
    api_key: str
    model: Optional[str] = None


class PipelineCompileRequest(BaseModel):
    description: str
    context: Optional[dict] = None


class PipelineRunRequest(BaseModel):
    pipeline_id: str
    context: dict = {}


# ---------------------------------------------------------------------------
# Dependency: these get set by app.py at startup
# ---------------------------------------------------------------------------

_gateway = None
_db = None
_skills = None
_registry = None
_surveillance = None
_shadow = None
_cibil = None
_recommendations = None
_model_registry = None
_pipeline_compiler = None
_pipeline_runner = None
_pipeline_store = None
_ws_connections: list[WebSocket] = []

# Next-Gen module instances
_cognitive_memory = None
_cognitive_graph = None
_connector_forge = None
_forge_registry = None
_evolution_engine = None
_knowledge_broker = None
_intent_mesh = None
_red_team = None
_nl_policy = None
_pre_executor = None
_causal_diagnosis = None
_marketplace = None

# Phase 4 module instances
_embedding_engine = None
_vector_store = None
_semantic_cache = None
_document_processor = None
_neural_bus = None
_retrieval_cortex = None
_template_registry = None
_template_generator = None
_deploy_pipeline = None


def set_dependencies(
    gateway, db, skills=None, *,
    registry=None, surveillance=None, shadow_engine=None,
    cibil_engine=None, recommendation_engine=None,
    model_registry=None,
    pipeline_compiler=None, pipeline_runner=None, pipeline_store=None,
    cognitive_memory=None, cognitive_graph_factory=None,
    connector_forge=None, forge_registry=None,
    evolution_engine=None, knowledge_broker=None,
    intent_mesh=None, red_team=None, nl_policy=None,
    pre_executor=None, causal_diagnosis=None, marketplace=None,
    embedding_engine=None, vector_store=None, semantic_cache=None,
    document_processor=None, neural_bus=None, retrieval_cortex=None,
    template_registry=None, template_generator=None, deploy_pipeline=None,
):
    """Called by app.py to wire up all modules."""
    global _gateway, _db, _skills, _registry, _surveillance, _shadow
    global _cibil, _recommendations, _model_registry
    global _pipeline_compiler, _pipeline_runner, _pipeline_store
    global _cognitive_memory, _cognitive_graph, _connector_forge, _forge_registry
    global _evolution_engine, _knowledge_broker, _intent_mesh
    global _red_team, _nl_policy, _pre_executor, _causal_diagnosis, _marketplace
    global _embedding_engine, _vector_store, _semantic_cache, _document_processor
    global _neural_bus, _retrieval_cortex, _template_registry, _template_generator, _deploy_pipeline
    _gateway = gateway
    _db = db
    _skills = skills
    _registry = registry
    _surveillance = surveillance
    _shadow = shadow_engine
    _cibil = cibil_engine
    _recommendations = recommendation_engine
    _model_registry = model_registry
    _pipeline_compiler = pipeline_compiler
    _pipeline_runner = pipeline_runner
    _pipeline_store = pipeline_store
    _cognitive_memory = cognitive_memory
    _cognitive_graph = cognitive_graph_factory
    _connector_forge = connector_forge
    _forge_registry = forge_registry
    _evolution_engine = evolution_engine
    _knowledge_broker = knowledge_broker
    _intent_mesh = intent_mesh
    _red_team = red_team
    _nl_policy = nl_policy
    _pre_executor = pre_executor
    _causal_diagnosis = causal_diagnosis
    _marketplace = marketplace
    _embedding_engine = embedding_engine
    _vector_store = vector_store
    _semantic_cache = semantic_cache
    _document_processor = document_processor
    _neural_bus = neural_bus
    _retrieval_cortex = retrieval_cortex
    _template_registry = template_registry
    _template_generator = template_generator
    _deploy_pipeline = deploy_pipeline


async def _broadcast(data: dict):
    """Broadcast to all WebSocket connections."""
    message = json.dumps(data, default=str)
    disconnected = []
    for ws in _ws_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_connections.remove(ws)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Top-level health check for Docker HEALTHCHECK and monitoring."""
    checks = {
        "gateway": _gateway is not None,
        "database": _db is not None,
        "model_registry": _model_registry is not None,
        "pipeline_engine": _pipeline_runner is not None,
    }
    all_ok = all(checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
        "version": "0.3.0",
    }


@router.post("/evaluate")
async def evaluate_action(req: EvaluateRequest):
    """Evaluate and execute a tool call through the security pipeline."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    result = await _gateway.handle_tool_call(
        agent_id=req.agent_id,
        session_id=req.session_id,
        tool_name=req.tool_name,
        parameters=req.parameters,
    )

    # Persist to DB
    if _db:
        audit_events = _gateway._audit.get_all()
        if audit_events:
            latest = audit_events[-1]
            await _db.save_audit_event(latest.model_dump(mode="json"))

    # Broadcast to dashboard
    await _broadcast({
        "type": "action",
        "data": result,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return result


@router.post("/log")
async def log_result(req: LogRequest):
    """Log a tool execution result."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    from agentvault.models import Decision, LogLevel

    event = _gateway._audit.log_action(
        agent_id=req.agent_id,
        session_id=req.session_id,
        trace_id=req.trace_id,
        action_name=req.action_name,
        tool_name=req.tool_name,
        parameters=req.parameters,
        decision=Decision(req.decision),
        reasoning=req.reasoning,
        result=req.result,
    )

    if _db:
        await _db.save_audit_event(event.model_dump(mode="json"))

    return {"status": "logged", "event_id": event.id}


@router.get("/audit")
async def query_audit(
    agent_id: Optional[str] = None,
    decision: Optional[str] = None,
    action_name: Optional[str] = None,
    session_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
):
    """Query the audit log with filters."""
    if _db:
        events = await _db.query_audit(
            agent_id=agent_id,
            decision=decision,
            action_name=action_name,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        return {"events": events, "count": len(events)}

    # Fallback to in-memory
    if _gateway:
        from agentvault.models import Decision as Dec
        events = _gateway._audit.query(
            agent_id=agent_id,
            decision=Dec(decision) if decision else None,
            action_name=action_name,
            limit=limit,
            offset=offset,
        )
        return {
            "events": [e.model_dump(mode="json") for e in events],
            "count": len(events),
        }

    return {"events": [], "count": 0}


@router.get("/audit/replay/{session_id}")
async def session_replay(session_id: str):
    """Get full session timeline for replay."""
    if _db:
        events = await _db.get_session_replay(session_id)
        return {"session_id": session_id, "events": events, "count": len(events)}

    if _gateway:
        events = _gateway._audit.get_session(session_id)
        return {
            "session_id": session_id,
            "events": [e.model_dump(mode="json") for e in events],
            "count": len(events),
        }

    return {"session_id": session_id, "events": [], "count": 0}


@router.get("/audit/verify")
async def verify_chain():
    """Verify hash chain integrity."""
    if _db:
        result = await _db.verify_chain()
        return result

    if _gateway:
        valid, break_idx = _gateway._audit.verify()
        return {
            "valid": valid,
            "count": _gateway._audit.count,
            "break_index": break_idx,
        }

    return {"valid": True, "count": 0, "break_index": None}


@router.get("/policies")
async def get_policies():
    """Get active policy rules."""
    if not _gateway:
        return {"policies": []}

    policies = _gateway._policy.policies
    return {
        "policies": [p.model_dump(mode="json") for p in policies],
        "rule_hit_counts": _gateway._policy.rule_hit_counts,
    }


@router.post("/policies/reload")
async def reload_policies():
    """Hot-reload policies from YAML file."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    reloaded = _gateway._policy.check_for_reload()
    if not reloaded:
        # Force reload
        _gateway._policy._reload()
        reloaded = True

    return {"reloaded": reloaded, "policies_count": len(_gateway._policy.policies)}


@router.get("/agents")
async def list_agents():
    """List agents with stats."""
    if _db:
        agents = await _db.get_agents()
        return {"agents": agents}

    return {"agents": []}


@router.get("/drift/alerts")
async def get_drift_alerts(
    agent_id: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    """Get drift alerts."""
    if _db:
        alerts = await _db.query_drift_alerts(
            agent_id=agent_id, level=level, limit=limit
        )
        return {"alerts": alerts, "count": len(alerts)}

    if _gateway:
        from agentvault.models import AlertLevel
        alerts = _gateway._drift.get_alerts(
            agent_id=agent_id,
            level=AlertLevel(level) if level else None,
            limit=limit,
        )
        return {
            "alerts": [a.model_dump(mode="json") for a in alerts],
            "count": len(alerts),
        }

    return {"alerts": [], "count": 0}


@router.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools."""
    if not _gateway:
        return {"tools": []}

    return {"tools": _gateway.list_tools()}


@router.get("/mcp/servers")
async def list_mcp_servers():
    """List connected MCP servers."""
    if not _gateway:
        return {"servers": []}

    return {"servers": _gateway.list_servers()}


@router.post("/escalations/{escalation_id}/resolve")
async def resolve_escalation(escalation_id: str, req: ResolveRequest):
    """Approve or reject a pending escalation."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    result = await _gateway.resolve_escalation(
        escalation_id=escalation_id,
        approved=req.approved,
        resolved_by=req.resolved_by,
    )

    if not result:
        raise HTTPException(404, f"Escalation {escalation_id} not found or already resolved")

    if _db:
        await _db.save_escalation(result)

    await _broadcast({
        "type": "escalation_resolved",
        "data": result,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return result


@router.get("/stats")
async def get_stats():
    """Get aggregate dashboard statistics."""
    if _db:
        stats = await _db.get_stats()
        # Add gateway-specific stats
        if _gateway:
            stats["connected_mcp_servers"] = len(_gateway.list_servers())
            stats["available_tools"] = len(_gateway.list_tools())
            stats["pending_escalations"] = len(_gateway.get_escalations("PENDING"))
        return stats

    if _gateway:
        return _gateway.get_stats()

    return {
        "total_actions": 0,
        "allowed": 0,
        "denied": 0,
        "escalated": 0,
        "drift_alerts": 0,
        "active_agents": 0,
        "chain_healthy": True,
    }


# ---------------------------------------------------------------------------
# Advanced Security Endpoints
# ---------------------------------------------------------------------------

@router.get("/security/status")
async def get_security_status():
    """Get comprehensive security overview across all modules."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")
    return _gateway.get_security_status()


@router.get("/honeypot/triggers")
async def get_honeypot_triggers(
    agent_id: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    """Get honeypot trigger history."""
    if not _gateway:
        return {"triggers": [], "count": 0}
    triggers = _gateway._honeypot.get_triggers(agent_id=agent_id, limit=limit)
    return {
        "triggers": [t.model_dump(mode="json") for t in triggers],
        "quarantined": _gateway._honeypot.get_quarantined_agents(),
        "count": len(triggers),
    }


@router.post("/honeypot/register")
async def register_honeypot(req: HoneypotRegisterRequest):
    """Register a new honeypot canary tool."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")
    _gateway._honeypot.register_canary(
        tool_name=req.tool_name,
        description=req.description,
        category=req.category,
    )
    return {"status": "registered", "tool": req.tool_name}


@router.get("/chain/violations")
async def get_chain_violations(
    agent_id: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    """Get chain analysis violations."""
    if not _gateway:
        return {"violations": [], "count": 0}
    violations = _gateway._chain.get_violations(agent_id=agent_id, limit=limit)
    return {
        "violations": [v.model_dump(mode="json") for v in violations],
        "count": len(violations),
    }


@router.get("/injection/alerts")
async def get_injection_alerts(
    agent_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    """Get prompt injection alerts."""
    if not _gateway:
        return {"alerts": [], "count": 0}
    alerts = _gateway._guard.get_alerts(agent_id=agent_id, limit=limit)
    if severity:
        from agentvault.models import AlertLevel
        alerts = [a for a in alerts if a.severity.value == severity.upper()]
    return {
        "alerts": [a.model_dump(mode="json") for a in alerts],
        "count": len(alerts),
    }


@router.get("/reputation")
async def get_reputation_scores():
    """Get agent reputation scores."""
    if not _gateway:
        return {"agents": []}
    scores = _gateway._reputation.get_all_scores()
    return {
        "agents": [s.model_dump(mode="json") for s in scores],
    }


@router.get("/reputation/{agent_id}")
async def get_agent_reputation(agent_id: str):
    """Get a specific agent's reputation details."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")
    score = _gateway._reputation.get_score(agent_id)
    if not score:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    return score.model_dump(mode="json")


@router.post("/simulate")
async def simulate_policy(req: SimulateRequest):
    """Run a policy simulation (dry-run)."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    from agentvault.policy import PolicyEngine
    from agentvault.simulator import PolicySimulator

    candidate = PolicyEngine()
    if not os.path.exists(req.candidate_policy_path):
        raise HTTPException(400, f"Policy file not found: {req.candidate_policy_path}")
    candidate.load(req.candidate_policy_path)

    events = req.audit_events
    if not events:
        # Use live audit events
        live_events = _gateway._audit.get_all()
        events = [e.model_dump(mode="json") for e in live_events]

    sim = PolicySimulator()
    result = sim.simulate(candidate, events)
    return result.model_dump(mode="json")


@router.get("/memory/violations")
async def get_memory_violations(
    agent_id: Optional[str] = None,
    limit: int = Query(50, le=200),
):
    """Get memory firewall violations."""
    if not _gateway:
        return {"violations": [], "count": 0}
    violations = _gateway._memory.get_violations(agent_id=agent_id, limit=limit)
    return {
        "violations": [v.model_dump(mode="json") for v in violations],
        "count": len(violations),
    }


@router.get("/narrative/session/{session_id}")
async def get_session_narrative(session_id: str):
    """Generate a natural language narrative for a session."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    from agentvault.narrative import NarrativeGenerator

    events = _gateway._audit.get_session(session_id)
    event_dicts = [e.model_dump(mode="json") for e in events]

    gen = NarrativeGenerator()
    narrative = gen.generate_session_narrative(event_dicts, session_id=session_id)
    return {"session_id": session_id, "narrative": narrative}


@router.get("/narrative/overview")
async def get_overview_narrative():
    """Generate an overview narrative across all agents."""
    if not _gateway:
        raise HTTPException(500, "Gateway not initialized")

    from agentvault.narrative import NarrativeGenerator

    events = _gateway._audit.get_all()
    event_dicts = [e.model_dump(mode="json") for e in events]

    gen = NarrativeGenerator()
    narrative = gen.generate_overview(event_dicts)
    return {"narrative": narrative}


# ---------------------------------------------------------------------------
# Config / Test Endpoints
# ---------------------------------------------------------------------------

class DatabaseTestRequest(BaseModel):
    uri: str

class PolicyValidateRequest(BaseModel):
    yaml_content: str

class SkillCreateRequest(BaseModel):
    name: str
    description: str
    version: str = "1.0"
    tags: list[str] = []
    author: str = "user"
    steps: list[dict]
    permissions: dict = {}
    enabled: bool = True

class SkillImportRequest(BaseModel):
    yaml_content: str


@router.post("/config/database/test")
async def test_database_connection(req: DatabaseTestRequest):
    """Test a database connection URI without applying it."""
    from server.database.factory import parse_uri, _import_class, _BUILTIN_SCHEMES
    import time

    try:
        scheme, cleaned = parse_uri(req.uri)
        if scheme not in _BUILTIN_SCHEMES:
            return {"ok": False, "error": f"Unsupported scheme: {scheme}"}

        provider_class = _import_class(_BUILTIN_SCHEMES[scheme])
        provider = provider_class(cleaned)
        await provider.connect()
        health = await provider.health_check()
        await provider.close()
        return {"ok": True, "scheme": scheme, "health": health}
    except ImportError as e:
        return {"ok": False, "error": f"Missing driver: {str(e)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/config/database/health")
async def get_database_health():
    """Current database health status."""
    if not _db:
        raise HTTPException(500, "Database not initialized")
    try:
        health = await _db.health_check()
        return {"ok": True, **health}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/config/system")
async def get_system_config():
    """Full system configuration status."""
    config = {
        "database": {"connected": _db is not None},
        "gateway": {"connected": _gateway is not None},
        "skills": {"loaded": _skills is not None},
        "modules": {},
    }
    if _gateway:
        config["gateway"]["tools"] = len(getattr(_gateway, '_tools', {}))
        config["gateway"]["servers"] = len(getattr(_gateway, '_servers', {}))
        for mod_name in ['_policy', '_audit', '_drift', '_honeypot', '_chain', '_guard', '_reputation', '_memory']:
            config["modules"][mod_name.lstrip('_')] = hasattr(_gateway, mod_name)
    if _db:
        try:
            health = await _db.health_check()
            config["database"]["health"] = health
        except Exception:
            config["database"]["health"] = {"ok": False}
    if _skills:
        config["skills"]["stats"] = _skills.get_stats()
    return config


@router.post("/config/policy/validate")
async def validate_policy(req: PolicyValidateRequest):
    """Validate a YAML policy without applying it."""
    import yaml as _yaml
    try:
        data = _yaml.safe_load(req.yaml_content)
        if not data or not isinstance(data, dict):
            return {"valid": False, "error": "Empty or invalid YAML"}
        if "rules" not in data:
            return {"valid": False, "error": "Missing 'rules' section"}
        rules = data["rules"]
        if not isinstance(rules, list):
            return {"valid": False, "error": "'rules' must be a list"}
        for i, rule in enumerate(rules):
            if "action" not in rule:
                return {"valid": False, "error": f"Rule {i+1} missing 'action'"}
            if "decision" not in rule:
                return {"valid": False, "error": f"Rule {i+1} missing 'decision'"}
            if rule["decision"] not in ("allow", "deny", "escalate"):
                return {"valid": False, "error": f"Rule {i+1} invalid decision: {rule['decision']}"}
        return {"valid": True, "rules_count": len(rules), "default": data.get("default", "deny")}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Skills CRUD Endpoints
# ---------------------------------------------------------------------------

@router.get("/skills")
async def list_skills():
    """List all registered skills."""
    if not _skills:
        return {"skills": [], "stats": {}}
    return {"skills": _skills.list_skills(), "stats": _skills.get_stats()}


@router.post("/skills")
async def create_skill(req: SkillCreateRequest):
    """Create a new skill."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    try:
        skill = _skills.create_skill(req.model_dump())
        return {"ok": True, "skill": skill.model_dump()}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/skills/{name}")
async def get_skill(name: str):
    """Get a skill by name."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    skill = _skills.get_skill(name)
    if not skill:
        raise HTTPException(404, f"Skill '{name}' not found")
    return {"skill": skill.model_dump()}


@router.put("/skills/{name}")
async def update_skill(name: str, req: SkillCreateRequest):
    """Update an existing skill."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    try:
        skill = _skills.update_skill(name, req.model_dump())
        return {"ok": True, "skill": skill.model_dump()}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/skills/{name}")
async def delete_skill(name: str):
    """Delete a skill."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    try:
        _skills.delete_skill(name)
        return {"ok": True, "deleted": name}
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/skills/{name}/execute")
async def execute_skill(name: str):
    """Get structured skill execution plan for an AI agent."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    try:
        plan = _skills.execute_skill(name)
        return plan
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/skills/import")
async def import_skill(req: SkillImportRequest):
    """Import a skill from YAML content."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    try:
        skill = _skills.import_skill(req.yaml_content)
        return {"ok": True, "skill": skill.model_dump()}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/skills/{name}/export")
async def export_skill(name: str):
    """Export a skill as YAML."""
    if not _skills:
        raise HTTPException(500, "Skills engine not initialized")
    try:
        yaml_str = _skills.export_skill(name)
        return {"yaml": yaml_str, "name": name}
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.get("/skills/audit")
async def get_skills_audit(limit: int = Query(50, ge=1, le=200)):
    """Get skill operations audit log."""
    if not _skills:
        return {"audit": []}
    return {"audit": _skills.get_audit_log(limit)}


# ---------------------------------------------------------------------------
# Model Registry Endpoints
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models(include_disabled: bool = False):
    """List all configured AI models (API keys redacted)."""
    if not _model_registry:
        return {"models": [], "stats": {}}
    return {
        "models": _model_registry.list_models(include_disabled=include_disabled),
        "stats": _model_registry.get_stats(),
    }


@router.post("/models")
async def add_model(req: ModelAddRequest):
    """
    Add a new AI model provider.
    
    The API key is encrypted before storage — it never appears in plaintext
    in the config file, API responses, or logs.
    """
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")

    model = _model_registry.add_model(
        name=req.name,
        provider=req.provider,
        model_id=req.model_id,
        api_key=req.api_key,
        base_url=req.base_url,
        is_default=req.is_default,
        task_categories=req.task_categories,
        priority=req.priority,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    # Return safe version (no encrypted key)
    safe = model.model_dump(mode="json")
    safe["has_api_key"] = bool(model.api_key_encrypted)
    safe.pop("api_key_encrypted", None)

    await _broadcast({
        "type": "model_added",
        "data": safe,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {"ok": True, "model": safe}


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model config (API key redacted)."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    model = _model_registry.get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    safe = model.model_dump(mode="json")
    safe["has_api_key"] = bool(model.api_key_encrypted)
    safe.pop("api_key_encrypted", None)
    return safe


@router.put("/models/{model_id}")
async def update_model(model_id: str, req: ModelUpdateRequest):
    """
    Update a model config.
    
    If api_key is provided, it replaces the old one (encrypted).
    Only provided fields are updated.
    """
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")

    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")

    model = _model_registry.update_model(model_id, **updates)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")

    safe = model.model_dump(mode="json")
    safe["has_api_key"] = bool(model.api_key_encrypted)
    safe.pop("api_key_encrypted", None)
    return {"ok": True, "model": safe}


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Remove a model from the registry."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")

    removed = _model_registry.remove_model(model_id)
    if not removed:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return {"ok": True, "deleted": model_id}


@router.get("/models/{model_id}/health")
async def check_model_health(model_id: str):
    """Run a health check on a specific model."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    result = await _model_registry.check_health(model_id)
    return result


@router.get("/models/health/all")
async def check_all_models_health():
    """Run health checks on all enabled models."""
    if not _model_registry:
        return {"results": []}
    results = await _model_registry.check_all_health()
    return {"results": results}


@router.get("/models/route/{task_category}")
async def get_model_for_task(task_category: str):
    """
    Get the best model for a task category.
    
    Categories: pipeline_design, code_generation, data_analysis,
    conversation, security, summarization, general, fast
    """
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    adapter = _model_registry.get_adapter_for_task(task_category)
    if not adapter:
        raise HTTPException(404, f"No model available for task '{task_category}'")
    return {"model": adapter.name, "task_category": task_category}


@router.post("/models/setup/openai")
async def quick_setup_openai(req: QuickSetupRequest):
    """Quick setup: add OpenAI with sensible defaults."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    model = _model_registry.quick_setup_openai(
        api_key=req.api_key,
        model=req.model or "gpt-4o",
    )
    safe = model.model_dump(mode="json")
    safe["has_api_key"] = True
    safe.pop("api_key_encrypted", None)
    return {"ok": True, "model": safe}


@router.post("/models/setup/anthropic")
async def quick_setup_anthropic(req: QuickSetupRequest):
    """Quick setup: add Anthropic Claude with sensible defaults."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    model = _model_registry.quick_setup_anthropic(
        api_key=req.api_key,
        model=req.model or "claude-sonnet-4-20250514",
    )
    safe = model.model_dump(mode="json")
    safe["has_api_key"] = True
    safe.pop("api_key_encrypted", None)
    return {"ok": True, "model": safe}


@router.post("/models/setup/gemini")
async def quick_setup_gemini(req: QuickSetupRequest):
    """Quick setup: add Google Gemini with sensible defaults."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    model = _model_registry.quick_setup_gemini(
        api_key=req.api_key,
        model=req.model or "gemini-2.0-flash",
    )
    safe = model.model_dump(mode="json")
    safe["has_api_key"] = True
    safe.pop("api_key_encrypted", None)
    return {"ok": True, "model": safe}


@router.post("/models/setup/ollama")
async def quick_setup_ollama(req: QuickSetupRequest):
    """Quick setup: add local Ollama model."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    model = _model_registry.quick_setup_ollama(
        model=req.model or "llama3.2",
    )
    safe = model.model_dump(mode="json")
    safe["has_api_key"] = False
    safe.pop("api_key_encrypted", None)
    return {"ok": True, "model": safe}


@router.post("/models/fallback")
async def set_fallback_chain(model_ids: list[str]):
    """Set the fallback chain — order of models to try if primary fails."""
    if not _model_registry:
        raise HTTPException(500, "Model registry not initialized")
    _model_registry.set_fallback_chain(model_ids)
    return {"ok": True, "chain": model_ids}


@router.get("/models/stats")
async def get_model_stats():
    """Get model registry statistics."""
    if not _model_registry:
        return {}
    return _model_registry.get_stats()


# ---------------------------------------------------------------------------
# Pipeline Engine Endpoints
# ---------------------------------------------------------------------------

@router.post("/pipelines/compile")
async def compile_pipeline(req: PipelineCompileRequest):
    """
    Compile a natural language description into a pipeline DAG.
    
    Uses LLM (via ModelRegistry) to generate the DAG, or falls back to
    built-in template matching if no LLM is available.
    """
    if not _pipeline_compiler:
        raise HTTPException(500, "Pipeline engine not initialized")

    spec = await _pipeline_compiler.compile(
        description=req.description,
        context=req.context,
    )

    # Auto-save to store
    if _pipeline_store:
        _pipeline_store.save(spec)

    await _broadcast({
        "type": "pipeline_compiled",
        "data": {"id": spec.id, "name": spec.name, "steps": len(spec.steps)},
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {"ok": True, "pipeline": spec.model_dump(mode="json")}


@router.post("/pipelines/run")
async def run_pipeline(req: PipelineRunRequest):
    """
    Execute a saved pipeline.
    
    Each tool_call step goes through MCPGateway's 12-step security pipeline.
    LLM calls go through ModelRegistry.
    """
    if not _pipeline_runner or not _pipeline_store:
        raise HTTPException(500, "Pipeline engine not initialized")

    pipeline = _pipeline_store.get(req.pipeline_id)
    if not pipeline:
        raise HTTPException(404, f"Pipeline '{req.pipeline_id}' not found")

    await _broadcast({
        "type": "pipeline_started",
        "data": {"id": pipeline.id, "name": pipeline.name},
        "timestamp": datetime.utcnow().isoformat(),
    })

    result = await _pipeline_runner.run(
        pipeline=pipeline,
        initial_context=req.context,
    )

    # Save updated pipeline state
    _pipeline_store.save(pipeline)

    await _broadcast({
        "type": "pipeline_completed",
        "data": result.model_dump(mode="json"),
        "timestamp": datetime.utcnow().isoformat(),
    })

    return result.model_dump(mode="json")


@router.get("/pipelines")
async def list_pipelines():
    """List all saved pipelines."""
    if not _pipeline_store:
        return {"pipelines": []}
    return {"pipelines": _pipeline_store.list_all()}


@router.get("/pipelines/templates")
async def list_pipeline_templates():
    """List built-in pipeline templates."""
    if not _pipeline_compiler:
        return {"templates": []}
    return {"templates": _pipeline_compiler.list_templates()}


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get a pipeline's full spec."""
    if not _pipeline_store:
        raise HTTPException(500, "Pipeline engine not initialized")
    pipeline = _pipeline_store.get(pipeline_id)
    if not pipeline:
        raise HTTPException(404, f"Pipeline '{pipeline_id}' not found")
    return pipeline.model_dump(mode="json")


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a saved pipeline."""
    if not _pipeline_store:
        raise HTTPException(500, "Pipeline engine not initialized")
    removed = _pipeline_store.delete(pipeline_id)
    if not removed:
        raise HTTPException(404, f"Pipeline '{pipeline_id}' not found")
    return {"ok": True, "deleted": pipeline_id}


@router.get("/pipelines/stats")
async def pipeline_stats():
    """Get pipeline health statistics for the CIBIL analytics dashboard."""
    if not _pipeline_store:
        return {"pipelines": [], "totals": {}}
    pipelines = _pipeline_store.list_all()
    stats = []
    total_runs = 0
    total_success = 0
    total_duration = 0
    for p in pipelines:
        runs = p.get("run_count", 0)
        successes = p.get("success_count", runs)  # Default: assume all succeeded
        avg_dur = p.get("avg_duration_ms", 0)
        success_rate = (successes / runs * 100) if runs > 0 else 0
        total_runs += runs
        total_success += successes
        total_duration += avg_dur * runs
        stats.append({
            "id": p.get("id", ""),
            "name": p.get("name", "Unnamed"),
            "runs": runs,
            "success_rate": round(success_rate, 1),
            "avg_duration_ms": round(avg_dur, 1),
            "step_count": len(p.get("steps", [])),
            "last_status": p.get("status", "draft"),
            "connectors_used": list(set(
                s.get("tool_name", "").split(".")[0]
                for s in p.get("steps", [])
                if s.get("type") == "tool_call" and s.get("tool_name")
            )),
        })
    return {
        "pipelines": stats,
        "totals": {
            "total_pipelines": len(pipelines),
            "total_runs": total_runs,
            "success_rate": round(total_success / total_runs * 100, 1) if total_runs > 0 else 0,
            "avg_duration_ms": round(total_duration / total_runs, 1) if total_runs > 0 else 0,
        }
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

async def websocket_endpoint(websocket: WebSocket):
    """Real-time event stream for the dashboard."""
    await websocket.accept()
    _ws_connections.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_connections))

    try:
        # Send initial stats
        if _db:
            stats = await _db.get_stats()
        elif _gateway:
            stats = _gateway.get_stats()
        else:
            stats = {}

        await websocket.send_text(json.dumps({
            "type": "init",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat(),
        }, default=str))

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_connections))


# ---------------------------------------------------------------------------
# Registry Endpoints
# ---------------------------------------------------------------------------

@router.get("/registry/connectors")
async def list_connectors(category: Optional[str] = None):
    """List available connector blueprints."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return {"connectors": _registry.list_connectors(category)}


@router.get("/registry/connectors/{connector_id}")
async def get_connector_spec(connector_id: str):
    """Get a specific connector blueprint."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    spec = _registry.get_connector_spec(connector_id)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_id}' not found")
    return spec.model_dump(mode="json")


@router.post("/registry/connect")
async def connect_service(req: ConnectRequest):
    """Set up a new connection using a connector spec."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    try:
        status = _registry.connect(
            connector_id=req.connector_id,
            config=req.config,
            instance_name=req.instance_name or None,
        )
        return {
            "success": True,
            "connection": status.model_dump(mode="json"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/registry/connections")
async def list_connections():
    """List all active connections."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return {"connections": _registry.list_connections()}


@router.delete("/registry/connections/{instance_id}")
async def disconnect_service(instance_id: str):
    """Disconnect a service instance."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    ok = _registry.disconnect(instance_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Connection '{instance_id}' not found")
    return {"success": True}


@router.get("/registry/health")
async def registry_health():
    """Run health checks on all connections."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    results = await _registry.health_check_all()
    return {"health": results}


@router.get("/registry/stats")
async def registry_stats():
    """Get registry-wide statistics."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return _registry.get_stats()


@router.get("/registry/suggestions")
async def registry_suggestions():
    """Get connector suggestions based on current setup."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    tools = [s.name for s in _registry._connections.values()]
    return {"suggestions": _registry.suggest_connectors(tools)}


# ---------------------------------------------------------------------------
# Surveillance Endpoints
# ---------------------------------------------------------------------------

@router.get("/surveillance/tools")
async def surveillance_tools(tool_name: Optional[str] = None):
    """Get tool response profiles and health stats."""
    if not _surveillance:
        raise HTTPException(status_code=503, detail="Surveillance not initialized")
    return {
        "tools": _surveillance.get_tool_stats(tool_name),
        "health": _surveillance.get_tool_health(),
    }


@router.get("/surveillance/anomalies")
async def surveillance_anomalies(
    tool_name: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    """Get flagged response anomalies."""
    if not _surveillance:
        raise HTTPException(status_code=503, detail="Surveillance not initialized")
    return {
        "anomalies": _surveillance.get_anomalies(tool_name, severity, limit)
    }


# ---------------------------------------------------------------------------
# Shadow Execution Endpoints
# ---------------------------------------------------------------------------

@router.get("/shadow/stats")
async def shadow_stats():
    """Get shadow execution engine statistics."""
    if not _shadow:
        raise HTTPException(status_code=503, detail="Shadow engine not initialized")
    return _shadow.get_stats()


@router.get("/shadow/results")
async def shadow_results(
    verdict: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    """Get recent shadow execution results."""
    if not _shadow:
        raise HTTPException(status_code=503, detail="Shadow engine not initialized")
    return {"results": _shadow.get_results(verdict, limit)}


# ---------------------------------------------------------------------------
# CIBIL Score Endpoints
# ---------------------------------------------------------------------------

@router.get("/cibil/models")
async def cibil_list_models():
    """List all tracked models with CIBIL scores."""
    if not _cibil:
        raise HTTPException(status_code=503, detail="CIBIL engine not initialized")
    return {"models": _cibil.list_models()}


@router.get("/cibil/models/{model_id}")
async def cibil_get_model(model_id: str):
    """Get full profile for a model."""
    if not _cibil:
        raise HTTPException(status_code=503, detail="CIBIL engine not initialized")
    profile = _cibil.get_profile(model_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return profile.model_dump(mode="json")


@router.get("/cibil/models/{model_id}/category/{category}")
async def cibil_category_detail(model_id: str, category: str):
    """Get detailed stats for a model in a specific category."""
    if not _cibil:
        raise HTTPException(status_code=503, detail="CIBIL engine not initialized")
    detail = _cibil.get_category_detail(model_id, category)
    if not detail:
        raise HTTPException(status_code=404, detail="Not found")
    return detail


@router.get("/cibil/models/{model_id}/report")
async def cibil_report_card(model_id: str):
    """Generate an exportable report card for a model."""
    if not _cibil:
        raise HTTPException(status_code=503, detail="CIBIL engine not initialized")
    card = _cibil.get_report_card(model_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return card.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Recommendation Endpoints
# ---------------------------------------------------------------------------

@router.get("/recommendations/models/{task_category}")
async def recommend_model(
    task_category: str,
    current_model: Optional[str] = None,
    top_n: int = Query(default=3, le=10),
):
    """Get model recommendations for a task category."""
    if not _recommendations:
        raise HTTPException(status_code=503, detail="Recommendations not initialized")
    recs = _recommendations.suggest_model(task_category, current_model, top_n)
    return {"recommendations": [r.model_dump(mode="json") for r in recs]}


@router.get("/recommendations/tools/{model_id}")
async def recommend_tools(
    model_id: str,
    task_category: Optional[str] = None,
    top_n: int = Query(default=5, le=20),
):
    """Get tool recommendations for a model."""
    if not _recommendations:
        raise HTTPException(status_code=503, detail="Recommendations not initialized")
    recs = _recommendations.suggest_tools(model_id, task_category, top_n)
    return {"recommendations": [r.model_dump(mode="json") for r in recs]}


@router.get("/recommendations/warnings/{model_id}")
async def get_warnings(
    model_id: str,
    task_category: Optional[str] = None,
):
    """Get warnings and known issues for a model."""
    if not _recommendations:
        raise HTTPException(status_code=503, detail="Recommendations not initialized")
    warnings = _recommendations.get_warnings(model_id, task_category)
    return {"warnings": [w.model_dump(mode="json") for w in warnings]}


@router.get("/recommendations/report/{model_id}")
async def model_report(model_id: str):
    """Full recommendation report for a model."""
    if not _recommendations:
        raise HTTPException(status_code=503, detail="Recommendations not initialized")
    return _recommendations.generate_report(model_id)


@router.get("/recommendations/dashboard")
async def recommendations_dashboard():
    """Dashboard summary of model scores and recommendations."""
    if not _recommendations:
        raise HTTPException(status_code=503, detail="Recommendations not initialized")
    return _recommendations.get_dashboard_summary()


# ---------------------------------------------------------------------------
# Next-Gen: Cognitive Graph
# ---------------------------------------------------------------------------

class CognitiveGraphRequest(BaseModel):
    goal: str
    agent_id: str = "default-agent"
    max_nodes: int = 50


@router.post("/cognitive/execute")
async def execute_cognitive_graph(req: CognitiveGraphRequest):
    """Execute a goal through the self-modifying cognitive graph."""
    if not _cognitive_graph:
        raise HTTPException(status_code=503, detail="Cognitive Graph not initialized")
    from agentvault.cognitive_graph import CognitiveGraph
    graph = CognitiveGraph(
        gateway=_gateway,
        model_registry=_model_registry,
        memory=_cognitive_memory,
        max_nodes=req.max_nodes,
    )
    result = await graph.execute(goal=req.goal, agent_id=req.agent_id)
    return {
        "status": result.status,
        "total_nodes": result.total_nodes,
        "completed": result.completed_nodes,
        "failed": result.failed_nodes,
        "mutations_applied": result.mutations_applied,
        "duration_ms": result.duration_ms,
        "final_output": result.final_output,
        "node_trace": result.node_trace,
    }


@router.get("/cognitive/memory/stats")
async def memory_stats():
    """Get cognitive memory statistics."""
    if not _cognitive_memory:
        raise HTTPException(status_code=503, detail="Cognitive Memory not initialized")
    return await _cognitive_memory.get_stats()


@router.post("/cognitive/memory/consolidate")
async def consolidate_memory():
    """Trigger memory consolidation (episodes → semantic rules)."""
    if not _cognitive_memory:
        raise HTTPException(status_code=503, detail="Cognitive Memory not initialized")
    adapter = _model_registry.get_adapter("reasoning") if _model_registry else None
    rules = await _cognitive_memory.consolidate(llm_adapter=adapter)
    return {"new_rules": len(rules), "rules": [r.rule for r in rules]}


# ---------------------------------------------------------------------------
# Next-Gen: Connector Forge
# ---------------------------------------------------------------------------

class ForgeRequest(BaseModel):
    spec_url: Optional[str] = None
    spec_json: Optional[dict] = None


@router.post("/forge/connectors")
async def forge_connector(req: ForgeRequest):
    """Forge a connector from an OpenAPI spec."""
    if not _connector_forge:
        raise HTTPException(status_code=503, detail="Connector Forge not initialized")
    if req.spec_url:
        connector = await _connector_forge.forge_from_url(req.spec_url)
    elif req.spec_json:
        connector = await _connector_forge.forge_from_spec(req.spec_json)
    else:
        raise HTTPException(status_code=400, detail="Provide spec_url or spec_json")
    if _forge_registry:
        await _forge_registry.save(connector)
    return {
        "id": connector.id,
        "name": connector.name,
        "actions": connector.list_actions(),
        "auth": connector.auth.type,
        "action_count": len(connector.actions),
    }


@router.get("/forge/connectors")
async def list_forged_connectors():
    """List all forged connectors."""
    if not _forge_registry:
        raise HTTPException(status_code=503, detail="Forge Registry not initialized")
    connectors = await _forge_registry.list_all()
    return {"connectors": [{"id": c.id, "name": c.name, "actions": len(c.actions), "health": c.health_score} for c in connectors]}


@router.get("/forge/connectors/{connector_id}")
async def get_forged_connector(connector_id: str):
    """Get details of a forged connector."""
    if not _forge_registry:
        raise HTTPException(status_code=503, detail="Forge Registry not initialized")
    c = await _forge_registry.load(connector_id)
    if not c:
        raise HTTPException(status_code=404, detail="Connector not found")
    return {"id": c.id, "name": c.name, "description": c.description, "actions": c.list_actions(), "health": c.health_score, "usage": c.usage_count}


# ---------------------------------------------------------------------------
# Next-Gen: Red Team Engine
# ---------------------------------------------------------------------------

class RedTeamRequest(BaseModel):
    agent_id: str = "default-agent"
    rounds: int = 30
    difficulty: float = 0.5


@router.post("/security/red-team")
async def run_red_team(req: RedTeamRequest):
    """Run an adversarial red team assessment."""
    if not _red_team:
        raise HTTPException(status_code=503, detail="Red Team Engine not initialized")
    report = await _red_team.run_assessment(
        agent_id=req.agent_id, rounds=req.rounds, difficulty=req.difficulty
    )
    return {
        "security_score": report.security_score,
        "total_attacks": report.total_attacks,
        "vulnerabilities": len(report.vulnerabilities),
        "critical": report.critical_count,
        "high": report.high_count,
        "category_breakdown": report.category_breakdown,
        "details": [{"severity": v.severity.value, "category": v.category.value, "description": v.description, "remediation": v.remediation} for v in report.vulnerabilities],
    }


# ---------------------------------------------------------------------------
# Next-Gen: Natural Language Policy
# ---------------------------------------------------------------------------

class NLPolicyRequest(BaseModel):
    rule: str
    author: str = "user"


@router.post("/policies/natural-language")
async def compile_nl_policy(req: NLPolicyRequest):
    """Compile a natural language security rule."""
    if not _nl_policy:
        raise HTTPException(status_code=503, detail="NL Policy Engine not initialized")
    policy = await _nl_policy.compile(req.rule, author=req.author)
    return {
        "id": policy.id,
        "name": policy.name,
        "action": policy.action.value,
        "triggers": policy.trigger_actions,
        "keywords": policy.trigger_keywords,
        "severity": policy.severity,
        "original_text": policy.original_text,
    }


@router.get("/policies/natural-language")
async def list_nl_policies():
    """List all compiled NL policies."""
    if not _nl_policy:
        raise HTTPException(status_code=503, detail="NL Policy Engine not initialized")
    policies = await _nl_policy.list_policies()
    return {"policies": [{"id": p.id, "name": p.name, "action": p.action.value, "original": p.original_text, "applied": p.applied_count, "blocked": p.blocked_count} for p in policies]}


# ---------------------------------------------------------------------------
# Next-Gen: Pre-Execution Simulator
# ---------------------------------------------------------------------------

class SimulateExecutionRequest(BaseModel):
    pipeline_id: Optional[str] = None
    steps: Optional[list[dict]] = None


@router.post("/simulate/pre-execution")
async def pre_execute_simulation(req: SimulateExecutionRequest):
    """Simulate a pipeline before executing it."""
    if not _pre_executor:
        raise HTTPException(status_code=503, detail="Pre-Execution Simulator not initialized")
    spec = {"name": req.pipeline_id or "adhoc", "steps": req.steps or []}
    if req.pipeline_id and _pipeline_store:
        stored = await _pipeline_store.get(req.pipeline_id)
        if stored:
            spec = stored
    report = await _pre_executor.simulate(spec)
    return {
        "recommendation": report.recommendation,
        "overall_risk": report.overall_risk,
        "estimated_duration_ms": report.estimated_duration_ms,
        "estimated_cost_usd": report.estimated_cost.total_usd,
        "api_calls": report.total_api_calls,
        "warnings": report.warnings,
        "failure_risks": [{"step": r.step_name, "type": r.risk_type, "impact": r.impact, "probability": r.probability, "mitigation": r.mitigation} for r in report.failure_risks],
    }


# ---------------------------------------------------------------------------
# Next-Gen: Agent Genetics
# ---------------------------------------------------------------------------

@router.get("/genetics/stats")
async def genetics_stats():
    """Get evolution engine statistics."""
    if not _evolution_engine:
        raise HTTPException(status_code=503, detail="Evolution Engine not initialized")
    return await _evolution_engine.get_stats()


@router.get("/genetics/champion")
async def get_champion():
    """Get the best-performing genome."""
    if not _evolution_engine:
        raise HTTPException(status_code=503, detail="Evolution Engine not initialized")
    champ = await _evolution_engine.get_champion()
    if not champ:
        return {"champion": None}
    from dataclasses import asdict
    return {"champion": asdict(champ)}


# ---------------------------------------------------------------------------
# Next-Gen: Knowledge Broker & Intent Mesh
# ---------------------------------------------------------------------------

@router.get("/mesh/topology")
async def mesh_topology():
    """Get Intent Mesh network topology."""
    if not _intent_mesh:
        raise HTTPException(status_code=503, detail="Intent Mesh not initialized")
    return await _intent_mesh.get_mesh_topology()


@router.get("/knowledge/stats")
async def knowledge_stats():
    """Get Knowledge Broker statistics."""
    if not _knowledge_broker:
        raise HTTPException(status_code=503, detail="Knowledge Broker not initialized")
    return await _knowledge_broker.get_stats()


# ---------------------------------------------------------------------------
# Next-Gen: Causal Diagnosis
# ---------------------------------------------------------------------------

class DiagnoseRequest(BaseModel):
    run_id: str
    goal: str = ""
    model: str = ""
    error_type: str = ""
    error_message: str = ""
    context_tokens: int = 0
    task_type: str = ""


@router.post("/diagnose")
async def diagnose_failure(req: DiagnoseRequest):
    """Diagnose why a pipeline run failed."""
    if not _causal_diagnosis:
        raise HTTPException(status_code=503, detail="Causal Diagnosis not initialized")
    diagnosis = await _causal_diagnosis.diagnose(req.model_dump())
    return {
        "primary_cause": diagnosis.primary_cause,
        "confidence": diagnosis.confidence,
        "similar_failures": diagnosis.similar_failures,
        "similar_successes": diagnosis.similar_successes,
        "causes": [{"variable": c.variable, "value": c.value, "contribution": c.contribution, "evidence": c.evidence} for c in diagnosis.causes],
        "fixes": [{"action": f.action, "type": f.fix_type, "improvement": f.estimated_improvement, "auto_applicable": f.auto_applicable} for f in diagnosis.fixes],
    }


# ---------------------------------------------------------------------------
# Next-Gen: Marketplace
# ---------------------------------------------------------------------------

@router.get("/marketplace/search")
async def marketplace_search(q: str = "", limit: int = Query(20, le=50)):
    """Search the connector marketplace."""
    if not _marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not initialized")
    results = await _marketplace.search(q, limit=limit)
    return {"results": [{"id": r.id, "name": r.name, "description": r.description, "health": r.health_score, "installs": r.total_installs, "categories": r.categories} for r in results]}


@router.get("/marketplace/featured")
async def marketplace_featured():
    """Get featured connectors."""
    if not _marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not initialized")
    results = await _marketplace.get_featured()
    return {"featured": [{"id": r.id, "name": r.name, "health": r.health_score, "installs": r.total_installs} for r in results]}


@router.get("/marketplace/stats")
async def marketplace_stats():
    """Get marketplace statistics."""
    if not _marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not initialized")
    return await _marketplace.get_stats()


# ---------------------------------------------------------------------------
# Next-Gen: Unified Dashboard Stats
# ---------------------------------------------------------------------------

@router.get("/next-gen/stats")
async def next_gen_stats():
    """Unified stats for all next-gen systems."""
    stats = {}
    if _cognitive_memory:
        stats["memory"] = await _cognitive_memory.get_stats()
    if _evolution_engine:
        stats["genetics"] = await _evolution_engine.get_stats()
    if _knowledge_broker:
        stats["knowledge"] = await _knowledge_broker.get_stats()
    if _intent_mesh:
        stats["mesh"] = await _intent_mesh.get_stats()
    if _nl_policy:
        stats["nl_policies"] = await _nl_policy.get_stats()
    if _causal_diagnosis:
        stats["diagnosis"] = await _causal_diagnosis.get_stats()
    if _marketplace:
        stats["marketplace"] = await _marketplace.get_stats()
    if _embedding_engine:
        stats["embeddings"] = await _embedding_engine.get_stats()
    if _vector_store:
        stats["vector_store"] = await _vector_store.get_stats()
    if _semantic_cache:
        stats["cache"] = await _semantic_cache.get_stats()
    if _document_processor:
        stats["documents"] = await _document_processor.get_stats()
    if _neural_bus:
        stats["neural_bus"] = await _neural_bus.get_stats()
    if _retrieval_cortex:
        stats["retrieval"] = await _retrieval_cortex.get_stats()
    return stats


# ===========================================================================
# Phase 4: Infrastructure & Feature Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@router.post("/embeddings/embed")
async def embed_text(request: Request):
    """Embed text into a vector."""
    if not _embedding_engine:
        raise HTTPException(503, "Embedding engine not configured")
    body = await request.json()
    text = body.get("text", "")
    result = await _embedding_engine.embed(text)
    return {
        "vector": result.vector[:10],  # Truncated for dashboard display
        "dimensions": result.dimensions,
        "model": result.model,
        "source": result.source,
        "full_dimensions": len(result.vector),
    }

@router.get("/embeddings/stats")
async def embedding_stats():
    if not _embedding_engine:
        raise HTTPException(503, "Embedding engine not configured")
    return await _embedding_engine.get_stats()


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

@router.post("/vectors/search")
async def vector_search(request: Request):
    """Search the vector store."""
    if not _vector_store or not _embedding_engine:
        raise HTTPException(503, "Vector store not configured")
    body = await request.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    collection = body.get("collection", "documents")
    embed_result = await _embedding_engine.embed(query)
    results = await _vector_store.search(
        query_embedding=embed_result.vector,
        collection=collection,
        top_k=top_k,
    )
    return {
        "query": query,
        "results": [{"text": r.text[:200], "score": r.score, "id": r.id, "metadata": r.metadata} for r in results],
        "total": len(results),
    }

@router.get("/vectors/stats")
async def vector_stats():
    if not _vector_store:
        raise HTTPException(503, "Vector store not configured")
    return await _vector_store.get_stats()


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------

@router.get("/cache/stats")
async def cache_stats():
    """Get semantic cache statistics."""
    if not _semantic_cache:
        raise HTTPException(503, "Semantic cache not configured")
    return await _semantic_cache.get_stats()


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------

@router.post("/documents/ingest")
async def ingest_document(request: Request):
    """Ingest a document into the vector store."""
    if not _document_processor:
        raise HTTPException(503, "Document processor not configured")
    body = await request.json()
    text = body.get("text", "")
    source_name = body.get("source_name", "unnamed")
    collection = body.get("collection", "documents")
    result = await _document_processor.ingest(text=text, source_name=source_name, collection=collection)
    return {
        "source_name": result.source_name,
        "chunks": result.total_chunks,
        "characters": result.total_characters,
        "model": result.embedding_model,
        "duration_ms": result.duration_ms,
    }

@router.get("/documents/stats")
async def document_stats():
    if not _document_processor:
        raise HTTPException(503, "Document processor not configured")
    return await _document_processor.get_stats()


# ---------------------------------------------------------------------------
# Neural Bus
# ---------------------------------------------------------------------------

@router.get("/bus/events")
async def bus_events(channel: str = None, limit: int = 50):
    """Get recent events from the neural bus."""
    if not _neural_bus:
        raise HTTPException(503, "Neural bus not configured")
    return {
        "events": _neural_bus.get_recent(channel=channel, limit=limit),
        "stats": await _neural_bus.get_stats(),
    }

@router.get("/bus/stats")
async def bus_stats():
    if not _neural_bus:
        raise HTTPException(503, "Neural bus not configured")
    return await _neural_bus.get_stats()


# ---------------------------------------------------------------------------
# Retrieval Cortex (RAG)
# ---------------------------------------------------------------------------

@router.post("/retrieve")
async def retrieve(request: Request):
    """Adaptive RAG retrieval."""
    if not _retrieval_cortex:
        raise HTTPException(503, "Retrieval cortex not configured")
    body = await request.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    strategy = body.get("strategy")  # Optional: override auto-selection
    from agentvault.retrieval_cortex import RetrievalStrategy
    strat = RetrievalStrategy(strategy) if strategy else None
    response = await _retrieval_cortex.retrieve(query=query, top_k=top_k, strategy=strat)
    return {
        "query": response.query,
        "strategy": response.strategy_used.value,
        "results": [{"text": r.text[:200], "score": r.score, "source": r.source} for r in response.results],
        "duration_ms": response.duration_ms,
        "from_cache": response.from_cache,
    }

@router.post("/retrieve/feedback")
async def retrieval_feedback(request: Request):
    """Provide feedback on retrieval quality."""
    if not _retrieval_cortex:
        raise HTTPException(503, "Retrieval cortex not configured")
    body = await request.json()
    from agentvault.retrieval_cortex import RetrievalStrategy, RetrievalResponse
    strategy = RetrievalStrategy(body.get("strategy", "dense"))
    positive = body.get("positive", True)
    await _retrieval_cortex.feedback(
        RetrievalResponse(strategy_used=strategy),
        positive=positive,
    )
    return {"status": "feedback_recorded"}

@router.get("/retrieve/stats")
async def retrieval_stats():
    if not _retrieval_cortex:
        raise HTTPException(503, "Retrieval cortex not configured")
    return await _retrieval_cortex.get_stats()


# ---------------------------------------------------------------------------
# Agent Templates
# ---------------------------------------------------------------------------

@router.post("/templates/generate")
async def generate_template(request: Request):
    """Generate an agent template from English description."""
    if not _template_generator:
        raise HTTPException(503, "Template generator not configured")
    body = await request.json()
    description = body.get("description", "")
    author = body.get("author", "user")
    template = await _template_generator.generate(description, author)
    if _template_registry:
        await _template_registry.save(template)
    from dataclasses import asdict
    return {"id": template.id, "config": asdict(template.config)}

@router.get("/templates")
async def list_templates():
    """List all agent templates."""
    if not _template_registry:
        raise HTTPException(503, "Template registry not configured")
    templates = await _template_registry.list_all()
    from dataclasses import asdict
    return [{"id": t.id, "name": t.config.name, "description": t.config.description, "tags": t.config.tags, "installs": t.installs} for t in templates]

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    if not _template_registry:
        raise HTTPException(503, "Template registry not configured")
    t = await _template_registry.load(template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    from dataclasses import asdict
    return {"id": t.id, "config": asdict(t.config), "installs": t.installs, "rating": t.rating}

@router.get("/templates/stats")
async def template_stats():
    if not _template_registry:
        raise HTTPException(503, "Template registry not configured")
    return await _template_registry.get_stats()


# ---------------------------------------------------------------------------
# Deploy Pipeline
# ---------------------------------------------------------------------------

@router.post("/deploy")
async def deploy_agent(request: Request):
    """Deploy an agent with full safety pipeline."""
    if not _deploy_pipeline:
        raise HTTPException(503, "Deploy pipeline not configured")
    body = await request.json()
    from agentvault.deploy_engine import DeploymentConfig
    config = DeploymentConfig(
        agent_id=body.get("agent_id", ""),
        version=body.get("version", "1.0.0"),
        canary_percent=body.get("canary_percent", 10.0),
        security_min_score=body.get("security_min_score", 0.7),
        auto_rollback=body.get("auto_rollback", True),
    )
    deployment = await _deploy_pipeline.deploy(config)
    return {
        "id": deployment.id,
        "stage": deployment.stage.value,
        "gates": [{"name": g.name, "passed": g.passed, "duration_ms": g.duration_ms} for g in deployment.gates],
        "canary_percent": deployment.current_canary_percent,
        "duration_ms": deployment.duration_ms,
        "error": deployment.error,
    }

@router.get("/deploy")
async def list_deployments():
    if not _deploy_pipeline:
        raise HTTPException(503, "Deploy pipeline not configured")
    return await _deploy_pipeline.list_deployments()

@router.get("/deploy/{deployment_id}")
async def get_deployment(deployment_id: str):
    if not _deploy_pipeline:
        raise HTTPException(503, "Deploy pipeline not configured")
    d = await _deploy_pipeline.get_deployment(deployment_id)
    if not d:
        raise HTTPException(404, "Deployment not found")
    return {
        "id": d.id,
        "stage": d.stage.value,
        "agent_id": d.config.agent_id,
        "gates": [{"name": g.name, "passed": g.passed, "result": g.result} for g in d.gates],
        "canary_percent": d.current_canary_percent,
        "rollback_reason": d.rollback_reason,
    }

@router.get("/deploy/stats")
async def deploy_stats():
    if not _deploy_pipeline:
        raise HTTPException(503, "Deploy pipeline not configured")
    return await _deploy_pipeline.get_stats()
