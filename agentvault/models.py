"""
AgentVault — Data Models
All Pydantic schemas used across the SDK, server, and dashboard.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    """Firewall decision outcome."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    ESCALATE = "ESCALATE"


class AlertLevel(str, Enum):
    """Drift alert severity."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class LogLevel(str, Enum):
    """Audit log verbosity."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"
    CRITICAL = "critical"


class EscalationStatus(str, Enum):
    """Escalation item status."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Core Action Models
# ---------------------------------------------------------------------------

class AgentAction(BaseModel):
    """An action that an agent wants to perform."""
    agent_id: str
    action_name: str
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Policy Models
# ---------------------------------------------------------------------------

class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    max: int
    window: int  # seconds


class SandboxRuleConfig(BaseModel):
    """Per-rule sandbox overrides."""
    timeout: int = 30
    memory_mb: int = 256
    read_only: bool = False
    allowed_paths: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)


class PolicyRule(BaseModel):
    """A single policy rule."""
    action: str  # exact, wildcard (read_*), or regex
    decision: Decision
    rate_limit: Optional[RateLimitConfig] = None
    sandbox: Optional[SandboxRuleConfig] = None
    log_level: LogLevel = LogLevel.STANDARD
    allowed_hours: Optional[str] = None  # e.g. "9-17"
    allowed_days: Optional[str] = None  # e.g. "mon-fri"
    conditions: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class PolicyConfig(BaseModel):
    """Full policy configuration loaded from YAML."""
    agent: str = "*"  # agent ID or wildcard
    default: Decision = Decision.DENY
    rules: list[PolicyRule] = Field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Firewall Decision Models
# ---------------------------------------------------------------------------

class FirewallDecision(BaseModel):
    """Result of evaluating an action through the firewall."""
    decision: Decision
    reasoning: str
    trace_id: str
    matched_rule: Optional[str] = None
    confidence_score: Optional[float] = None
    drift_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == Decision.DENY

    @property
    def escalated(self) -> bool:
        return self.decision == Decision.ESCALATE


# ---------------------------------------------------------------------------
# Audit Models
# ---------------------------------------------------------------------------

class AuditEvent(BaseModel):
    """An immutable audit log entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    session_id: str
    trace_id: str
    action_name: str
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    decision: Decision
    reasoning: str
    confidence_score: Optional[float] = None
    drift_score: Optional[float] = None
    input_hash: str = ""
    output_hash: str = ""
    result: Optional[dict[str, Any]] = None
    log_level: LogLevel = LogLevel.STANDARD
    event_hash: str = ""
    previous_hash: str = ""


# ---------------------------------------------------------------------------
# Drift Models
# ---------------------------------------------------------------------------

class DriftAlert(BaseModel):
    """A behavioral drift alert."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    deviation_score: float
    alert_level: AlertLevel
    baseline_distribution: dict[str, float] = Field(default_factory=dict)
    current_distribution: dict[str, float] = Field(default_factory=dict)
    window: str = "1h"  # time window used
    message: str = ""


# ---------------------------------------------------------------------------
# Confidence Models
# ---------------------------------------------------------------------------

class ConfidenceFactor(BaseModel):
    """A single factor contributing to confidence score."""
    name: str
    score: float  # 0.0–1.0
    detail: str = ""


class ConfidenceScore(BaseModel):
    """LLM output confidence assessment."""
    value: float = Field(ge=0.0, le=1.0)
    factors: list[ConfidenceFactor] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Sandbox Models
# ---------------------------------------------------------------------------

class SandboxConfig(BaseModel):
    """Sandbox execution configuration."""
    allowed_paths: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 256
    read_only: bool = False
    scan_pii: bool = True


class SandboxViolation(BaseModel):
    """A sandbox violation record."""
    violation_type: str  # path, domain, timeout, memory, pii
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# MCP Models
# ---------------------------------------------------------------------------

class MCPToolInfo(BaseModel):
    """Information about a tool from an MCP server."""
    name: str
    description: str = ""
    server_id: str = ""
    parameters_schema: dict[str, Any] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    """Configuration for a downstream MCP server."""
    server_id: str
    name: str
    url: str
    api_key: Optional[str] = None
    enabled: bool = True
    tools: list[MCPToolInfo] = Field(default_factory=list)


class MCPToolCall(BaseModel):
    """A tool call routed through the MCP gateway."""
    tool_name: str
    server_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Escalation Models
# ---------------------------------------------------------------------------

class EscalationItem(BaseModel):
    """An action waiting for human approval."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    session_id: str
    trace_id: str
    action_name: str
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    status: EscalationStatus = EscalationStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Dashboard / Stats Models
# ---------------------------------------------------------------------------

class DashboardStats(BaseModel):
    """Aggregate statistics for the dashboard."""
    total_actions: int = 0
    allowed: int = 0
    denied: int = 0
    escalated: int = 0
    drift_alerts: int = 0
    chain_healthy: bool = True
    active_agents: int = 0
    connected_mcp_servers: int = 0


class AgentStats(BaseModel):
    """Per-agent statistics."""
    agent_id: str
    total_actions: int = 0
    allowed: int = 0
    denied: int = 0
    escalated: int = 0
    last_active: Optional[datetime] = None
    action_distribution: dict[str, int] = Field(default_factory=dict)
    drift_alert_count: int = 0


# ---------------------------------------------------------------------------
# Trust / Reputation Models
# ---------------------------------------------------------------------------

class TrustLevel(str, Enum):
    """Agent trust tier based on behavioral history."""
    UNTRUSTED = "UNTRUSTED"    # New / quarantined agents
    LIMITED = "LIMITED"        # Low trust, restricted access
    STANDARD = "STANDARD"     # Default tier
    TRUSTED = "TRUSTED"       # Proven good behavior


class ReputationScore(BaseModel):
    """Per-agent reputation tracking."""
    agent_id: str
    score: float = Field(default=50.0, ge=0.0, le=100.0)
    trust_level: TrustLevel = TrustLevel.STANDARD
    total_actions: int = 0
    violations: int = 0
    clean_streak: int = 0
    last_violation: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    history: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Honeypot Models
# ---------------------------------------------------------------------------

class HoneypotAlert(BaseModel):
    """Triggered when an agent touches a canary tool."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    agent_id: str
    session_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    quarantined: bool = True


# ---------------------------------------------------------------------------
# Chain Analysis Models
# ---------------------------------------------------------------------------

class ChainViolation(BaseModel):
    """Violation from forbidden action sequence detection."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    session_id: str
    matched_pattern: list[str]
    actual_sequence: list[str]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Prompt Injection Models
# ---------------------------------------------------------------------------

class InjectionAlert(BaseModel):
    """Alert from prompt injection detection in tool parameters."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    session_id: str
    tool_name: str
    parameter_key: str
    matched_pattern: str
    severity: AlertLevel
    snippet: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Simulation Models
# ---------------------------------------------------------------------------

class SimulationResult(BaseModel):
    """Result of replaying audit events through a candidate policy."""
    total_events: int = 0
    flipped_decisions: list[dict] = Field(default_factory=list)
    impact_summary: dict[str, int] = Field(default_factory=dict)
    affected_agents: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory Firewall Models
# ---------------------------------------------------------------------------

class MemoryViolation(BaseModel):
    """Detected cross-session data smuggling attempt."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    current_session: str
    source_session: str
    fingerprint: str
    similarity: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Tool / Service Registry Models
# ---------------------------------------------------------------------------

class ConnectorCategory(str, Enum):
    """Categories for tool/service connectors."""
    DATABASE = "database"
    MESSAGING = "messaging"
    CLOUD = "cloud"
    DEV_TOOLS = "dev_tools"
    API = "api"
    STORAGE = "storage"
    MONITORING = "monitoring"
    AI_MODEL = "ai_model"


class ConnectorSpec(BaseModel):
    """Definition for a tool/service connector."""
    connector_id: str
    name: str
    category: ConnectorCategory
    description: str = ""
    icon: str = "🔌"
    config_schema: dict[str, Any] = Field(default_factory=dict)
    setup_steps: list[str] = Field(default_factory=list)
    health_check_endpoint: str = ""
    default_port: Optional[int] = None
    required_env_vars: list[str] = Field(default_factory=list)
    documentation_url: str = ""
    tags: list[str] = Field(default_factory=list)


class ConnectorStatus(BaseModel):
    """Live status of a connected tool/service."""
    connector_id: str
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: ConnectorCategory
    connected: bool = False
    healthy: bool = False
    last_health_check: Optional[datetime] = None
    latency_ms: float = 0.0
    error_count: int = 0
    total_calls: int = 0
    uptime_percent: float = 100.0
    config: dict[str, Any] = Field(default_factory=dict)
    connected_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Bidirectional Surveillance Models
# ---------------------------------------------------------------------------

class ResponseProfile(BaseModel):
    """Baseline profile for a tool's typical responses."""
    tool_name: str
    avg_latency_ms: float = 0.0
    avg_response_size: int = 0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_observations: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ResponseAnomaly(BaseModel):
    """Flagged suspicious tool response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    anomaly_type: str  # latency_spike, size_spike, error_burst, content_suspect
    detail: str
    severity: AlertLevel = AlertLevel.LOW
    observed_value: float = 0.0
    expected_value: float = 0.0
    deviation_factor: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Shadow Execution Models
# ---------------------------------------------------------------------------

class ShadowResult(BaseModel):
    """Result of pre-commit shadow execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    shadow_output: Any = None
    impact_score: float = 0.0  # 0.0 = safe, 1.0 = destructive
    files_affected: int = 0
    rows_affected: int = 0
    bytes_modified: int = 0
    side_effects: list[str] = Field(default_factory=list)
    verdict: str = "proceed"  # proceed, warn, escalate, block
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ImpactAssessment(BaseModel):
    """Damage estimate from shadow execution."""
    destructive: bool = False
    reversible: bool = True
    blast_radius: str = "none"  # none, local, service, global
    affected_resources: list[str] = Field(default_factory=list)
    risk_score: float = 0.0
    recommendation: str = "proceed"


# ---------------------------------------------------------------------------
# Model CIBIL Score Models
# ---------------------------------------------------------------------------

class WorkCategory(str, Enum):
    """Categories of work that models perform."""
    OPERATIONS = "operations"
    DEVELOPMENT = "development"
    RESEARCH = "research"
    CREATIVE = "creative"
    COMMUNICATION = "communication"
    DATA_ANALYSIS = "data_analysis"


class CategoryProfile(BaseModel):
    """Behavioral stats for a model in a specific work category."""
    category: WorkCategory
    tools_used: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    failure_rate: float = 0.0
    risk_incidents: int = 0
    total_actions: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    score: float = 50.0  # 0-100 category score


class ModelProfile(BaseModel):
    """Full cross-user behavioral profile for a model."""
    model_id: str
    overall_score: float = Field(default=50.0, ge=0.0, le=100.0)
    total_actions: int = 0
    total_instances: int = 0  # how many Shield instances report on this model
    category_profiles: dict[str, CategoryProfile] = Field(default_factory=dict)
    known_issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ModelReportCard(BaseModel):
    """Exportable summary of a model's behavioral profile."""
    model_id: str
    overall_score: float
    grade: str = "B"  # A, B, C, D, F
    total_actions: int = 0
    total_instances: int = 0
    best_categories: list[str] = Field(default_factory=list)
    weakest_categories: list[str] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Recommendation Models
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    """A model/tool/config suggestion based on CIBIL data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rec_type: str  # model, tool, config, warning
    title: str
    detail: str
    confidence: float = 0.0  # 0-1 how confident we are in this rec
    source_data_points: int = 0  # how many data points back this up
    model_id: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
