"""
AgentVault — Intelligent Middleware Platform for Autonomous Systems

Every MCP tool call. Governed. Audited. Sandboxed. Scored.

Modules:
    Core:       PolicyEngine, AuditChain, DriftDetector, ConfidenceScorer, ToolSandbox, MCPGateway
    Security:   HoneypotManager, ChainAnalyzer, PromptGuard, MemoryFirewall
    Trust:      ReputationEngine, CIBILEngine, RecommendationEngine
    Platform:   ServiceRegistry, ResponseSurveillance, ShadowEngine
"""

__version__ = "0.3.0"

from .sdk import VaultClient, VaultProtector, VaultAgent
from .adapters import (
    BaseLLMAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    OllamaAdapter,
    get_adapter,
    auto_detect_adapter,
)
from .models import (
    AgentAction,
    Decision,
    FirewallDecision,
    AuditEvent,
    DriftAlert,
    SandboxConfig,
    PolicyConfig,
    PolicyRule,
    TrustLevel,
    HoneypotAlert,
    ChainViolation,
    InjectionAlert,
    ReputationScore,
    SimulationResult,
    MemoryViolation,
    # Platform models
    ConnectorCategory,
    ConnectorSpec,
    ConnectorStatus,
    ResponseProfile,
    ResponseAnomaly,
    ShadowResult,
    ImpactAssessment,
    WorkCategory,
    CategoryProfile,
    ModelProfile,
    ModelReportCard,
    Recommendation,
)
from .policy import PolicyEngine
from .audit import AuditChain
from .drift import DriftDetector
from .confidence import ConfidenceScorer
from .sandbox import ToolSandbox
from .mcp_gateway import MCPGateway

# Advanced Security Modules
from .honeypot import HoneypotManager
from .chain_analyzer import ChainAnalyzer
from .prompt_guard import PromptGuard
from .reputation import ReputationEngine
from .simulator import PolicySimulator
from .memory_firewall import MemoryFirewall
from .narrative import NarrativeGenerator

# Platform Modules
from .registry import ServiceRegistry
from .surveillance import ResponseSurveillance
from .shadow import ShadowEngine
from .cibil import CIBILEngine
from .recommendations import RecommendationEngine
from .model_registry import ModelRegistry, ModelProvider, TaskCategory, ModelConfig
from .pipeline import (
    PipelineCompiler,
    PipelineRunner,
    PipelineStore,
    PipelineSpec,
    PipelineStep,
    PipelineRunResult,
    PipelineStatus,
    StepType,
    StepStatus,
)
from .connectors import (
    ConnectorExecutor,
    GitHubConnector,
    SlackConnector,
    PostgreSQLConnector,
    EmailConnector,
    S3Connector,
    BaseConnector,
)

# Next-Gen Modules
from .cognitive_memory import CognitiveMemoryManager, WorkingMemory, EpisodicMemory, SemanticMemory
from .cognitive_graph import CognitiveGraph, GraphNode, GraphMutation, MutationType, NodeStatus
from .connector_forge import ConnectorForge, ForgeRegistry, ForgedConnector, ForgedConnectorExecutor
from .agent_genetics import AgentGenome, EvolutionEngine, FitnessEvaluator
from .knowledge_broker import KnowledgeBroker
from .intent_mesh import IntentMesh, AgentManifest, CollaborationContract
from .red_team import RedTeamEngine, AttackCategory, VulnerabilityReport
from .nl_policy import NLPolicyCompiler, CompiledPolicy, PolicyAction
from .pre_executor import PreExecutionSimulator, PreExecutionReport
from .causal_diagnosis import CausalDiagnosisEngine, Diagnosis
from .marketplace import ConnectorMarketplace, MarketplaceListing

# Phase 4: Infrastructure
from .embedding_engine import EmbeddingEngine, LocalEmbedder, CloudEmbedder, CrossEncoderReranker
from .vector_store import VectorStore, VectorDocument, SimilarityResult
from .semantic_cache import SemanticCache, ExactCache, NearMatchCache
from .document_processor import DocumentProcessor, RecursiveChunker, ChunkConfig

# Phase 4: Features
from .neural_bus import NeuralBus, BusEvent, EventChannel
from .retrieval_cortex import RetrievalCortex, RetrievalStrategy, RetrievalResponse
from .agent_templates import AgentTemplate, TemplateGenerator, TemplateRegistry, TemplateConfig
from .deploy_engine import DeploymentPipeline, DeploymentConfig, Deployment, DeployStage

# Convenient module-level protector
vault = VaultProtector.__new__(VaultProtector)

__all__ = [
    # SDK
    "VaultClient",
    "VaultProtector",
    "VaultAgent",
    "vault",
    # Adapters
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "get_adapter",
    "auto_detect_adapter",
    # Core
    "PolicyEngine",
    "AuditChain",
    "DriftDetector",
    "ConfidenceScorer",
    "ToolSandbox",
    "MCPGateway",
    # Advanced Security
    "HoneypotManager",
    "ChainAnalyzer",
    "PromptGuard",
    "ReputationEngine",
    "PolicySimulator",
    "MemoryFirewall",
    "NarrativeGenerator",
    # Platform
    "ServiceRegistry",
    "ResponseSurveillance",
    "ShadowEngine",
    "CIBILEngine",
    "RecommendationEngine",
    # Model Registry
    "ModelRegistry",
    "ModelProvider",
    "TaskCategory",
    "ModelConfig",
    # Pipeline Engine
    "PipelineCompiler",
    "PipelineRunner",
    "PipelineStore",
    "PipelineSpec",
    "PipelineStep",
    "PipelineRunResult",
    "PipelineStatus",
    "StepType",
    "StepStatus",
    # Connectors
    "ConnectorExecutor",
    "GitHubConnector",
    "SlackConnector",
    "PostgreSQLConnector",
    "EmailConnector",
    "S3Connector",
    "BaseConnector",
    # Models
    "AgentAction",
    "Decision",
    "FirewallDecision",
    "AuditEvent",
    "DriftAlert",
    "SandboxConfig",
    "PolicyConfig",
    "PolicyRule",
    "TrustLevel",
    "HoneypotAlert",
    "ChainViolation",
    "InjectionAlert",
    "ReputationScore",
    "SimulationResult",
    "MemoryViolation",
    "ConnectorCategory",
    "ConnectorSpec",
    "ConnectorStatus",
    "ResponseProfile",
    "ResponseAnomaly",
    "ShadowResult",
    "ImpactAssessment",
    "WorkCategory",
    "CategoryProfile",
    "ModelProfile",
    "ModelReportCard",
    "Recommendation",
    # Next-Gen: Cognitive Memory
    "CognitiveMemoryManager",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    # Next-Gen: Cognitive Graph
    "CognitiveGraph",
    "GraphNode",
    "GraphMutation",
    "MutationType",
    "NodeStatus",
    # Next-Gen: Connector Forge
    "ConnectorForge",
    "ForgeRegistry",
    "ForgedConnector",
    "ForgedConnectorExecutor",
    # Next-Gen: Agent Genetics
    "AgentGenome",
    "EvolutionEngine",
    "FitnessEvaluator",
    # Next-Gen: Knowledge Broker
    "KnowledgeBroker",
    # Next-Gen: Intent Mesh
    "IntentMesh",
    "AgentManifest",
    "CollaborationContract",
    # Next-Gen: Red Team
    "RedTeamEngine",
    "AttackCategory",
    "VulnerabilityReport",
    # Next-Gen: NL Policy
    "NLPolicyCompiler",
    "CompiledPolicy",
    "PolicyAction",
    # Next-Gen: Pre-Execution Simulator
    "PreExecutionSimulator",
    "PreExecutionReport",
    # Next-Gen: Causal Diagnosis
    "CausalDiagnosisEngine",
    "Diagnosis",
    # Next-Gen: Marketplace
    "ConnectorMarketplace",
    "MarketplaceListing",
    # Phase 4: Infrastructure
    "EmbeddingEngine",
    "LocalEmbedder",
    "CloudEmbedder",
    "CrossEncoderReranker",
    "VectorStore",
    "VectorDocument",
    "SimilarityResult",
    "SemanticCache",
    "ExactCache",
    "NearMatchCache",
    "DocumentProcessor",
    "RecursiveChunker",
    "ChunkConfig",
    # Phase 4: Features
    "NeuralBus",
    "BusEvent",
    "EventChannel",
    "RetrievalCortex",
    "RetrievalStrategy",
    "RetrievalResponse",
    "AgentTemplate",
    "TemplateGenerator",
    "TemplateRegistry",
    "TemplateConfig",
    "DeploymentPipeline",
    "DeploymentConfig",
    "Deployment",
    "DeployStage",
]
