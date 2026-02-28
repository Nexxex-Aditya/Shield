"""
AgentVault — FastAPI Application
Intelligent Middleware Platform for Autonomous Systems.
Main server entry point with CORS, static files, startup hooks, and MCP gateway.
"""

from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent dir to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.policy import PolicyEngine
from agentvault.audit import AuditChain
from agentvault.drift import DriftDetector
from agentvault.confidence import ConfidenceScorer
from agentvault.mcp_gateway import MCPGateway
from agentvault.models import SandboxConfig
from agentvault.honeypot import HoneypotManager
from agentvault.chain_analyzer import ChainAnalyzer
from agentvault.prompt_guard import PromptGuard
from agentvault.reputation import ReputationEngine
from agentvault.memory_firewall import MemoryFirewall
from agentvault.skills import SkillsEngine

# Platform modules
from agentvault.registry import ServiceRegistry
from agentvault.surveillance import ResponseSurveillance
from agentvault.shadow import ShadowEngine
from agentvault.cibil import CIBILEngine
from agentvault.recommendations import RecommendationEngine
from agentvault.model_registry import ModelRegistry
from agentvault.pipeline import PipelineCompiler, PipelineRunner, PipelineStore
from agentvault.connectors import ConnectorExecutor

from server.database import DatabaseStore
from server.routes import router, set_dependencies, websocket_endpoint
from server.auth import auth_router, set_auth_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agentvault.app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLICY_PATH = os.environ.get("AGENTVAULT_POLICY", "policies/default.yaml")
DB_URI = os.environ.get("AGENTVAULT_DB", "agentvault.db")
SKILLS_DIR = os.environ.get("AGENTVAULT_SKILLS", "skills")
CONNECTORS_DIR = os.environ.get("AGENTVAULT_CONNECTORS", "connectors")
CONFIG_DIR = os.environ.get("AGENTVAULT_CONFIG", "config")
PIPELINES_DIR = os.environ.get("AGENTVAULT_PIPELINES", "pipelines")
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")

# ---------------------------------------------------------------------------
# Core Components
# ---------------------------------------------------------------------------

policy_engine = PolicyEngine()
audit_chain = AuditChain()
drift_detector = DriftDetector()
confidence_scorer = ConfidenceScorer()

sandbox_config = SandboxConfig(
    allowed_paths=[
        os.path.abspath("data"),
        os.path.abspath("examples"),
    ],
    allowed_domains=["localhost", "127.0.0.1"],
    max_execution_time=30,
    max_memory_mb=256,
    scan_pii=True,
)

# Advanced security modules
honeypot = HoneypotManager(auto_quarantine=True)
chain_analyzer = ChainAnalyzer(max_history=500)
prompt_guard = PromptGuard()
reputation = ReputationEngine(initial_score=50.0)
memory_firewall = MemoryFirewall()

# Platform modules
registry = ServiceRegistry(connectors_dir=CONNECTORS_DIR)
surveillance = ResponseSurveillance()
shadow_engine = ShadowEngine()
cibil_engine = CIBILEngine()
recommendation_engine = RecommendationEngine(cibil=cibil_engine)

# Model Registry (encrypted API key storage + routing)
model_registry = ModelRegistry(config_dir=CONFIG_DIR)

gateway = MCPGateway(
    policy_engine=policy_engine,
    audit_chain=audit_chain,
    drift_detector=drift_detector,
    confidence_scorer=confidence_scorer,
    sandbox_config=sandbox_config,
    honeypot=honeypot,
    chain_analyzer=chain_analyzer,
    prompt_guard=prompt_guard,
    reputation=reputation,
    memory_firewall=memory_firewall,
    shadow_engine=shadow_engine,
    surveillance=surveillance,
)

# Pipeline Engine (created after gateway so it can reference it)
pipeline_store = PipelineStore(pipelines_dir=PIPELINES_DIR)
pipeline_compiler = PipelineCompiler(model_registry=model_registry)
connector_executor = ConnectorExecutor()
pipeline_runner = PipelineRunner(
    gateway=gateway,
    model_registry=model_registry,
    connector_executor=connector_executor,
)

db_store = DatabaseStore(DB_URI)
skills_engine = SkillsEngine(skills_dir=SKILLS_DIR)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentVault",
    description="Intelligent Middleware Platform for Autonomous Systems",
    version="0.3.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API
app.include_router(router)

# Auth
app.include_router(auth_router, prefix="/api/v1")

# WebSocket
app.add_api_websocket_route("/ws/live", websocket_endpoint)


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("  AgentVault — Intelligent Middleware Platform")
    logger.info("  Starting up...")
    logger.info("=" * 60)

    # Init database (auto-detects backend from URI scheme)
    await db_store.connect()
    logger.info("Database connected: %s", DB_URI)

    # Wire auth module to DB
    set_auth_db(db_store)

    # Load policies
    if os.path.exists(POLICY_PATH):
        policy_engine.load(POLICY_PATH)
        logger.info("Policies loaded: %s", POLICY_PATH)
    else:
        logger.warning("Policy file not found: %s", POLICY_PATH)

    # Load skills
    loaded = skills_engine.load_directory()
    logger.info("Skills loaded: %d from %s", loaded, SKILLS_DIR)

    # Load connectors registry
    builtin_count = registry.load_builtins()
    custom_count = registry.load_custom()
    logger.info("Registry: %d built-in + %d custom connectors", builtin_count, custom_count)

    # Load model registry (saved model configs with encrypted keys)
    model_registry.load()
    model_count = len(model_registry.list_models())
    logger.info("Model registry: %d configured models", model_count)

    # Wire up dependencies (pass all platform modules)
    set_dependencies(
        gateway, db_store, skills_engine,
        registry=registry,
        surveillance=surveillance,
        shadow_engine=shadow_engine,
        cibil_engine=cibil_engine,
        recommendation_engine=recommendation_engine,
        model_registry=model_registry,
        pipeline_compiler=pipeline_compiler,
        pipeline_runner=pipeline_runner,
        pipeline_store=pipeline_store,
    )

    # Set up audit -> DB sync listener
    def on_audit_event(event):
        """Persist audit events to DB in background."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                db_store.save_audit_event(event.model_dump(mode="json"))
            )
        except RuntimeError:
            # No running event loop (shouldn't happen in production)
            pass
        except Exception as e:
            logger.error("Failed to persist audit event: %s", e)

    audit_chain.add_listener(on_audit_event)

    logger.info("AgentVault ready at http://localhost:8000")
    logger.info("Dashboard: http://localhost:8000")
    logger.info("API docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await db_store.close()
    logger.info("AgentVault shutdown complete")


# ---------------------------------------------------------------------------
# Dashboard static files
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve landing page."""
    landing = os.path.join(DASHBOARD_DIR, "landing.html")
    if os.path.exists(landing):
        return FileResponse(landing)
    # Fallback to dashboard
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Shield Command API is running. Dashboard not found."}


@app.get("/dashboard")
async def dashboard():
    """Serve dashboard app."""
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Dashboard not found."}


# Mount static files after routes to avoid path conflicts
if os.path.exists(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")

