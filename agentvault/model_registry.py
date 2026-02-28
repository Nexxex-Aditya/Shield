"""
AgentVault — Model Registry & Configuration

Centralised system for managing AI model providers, API keys, and routing.
This is the layer users interact with to configure which models power their
workflows, pipelines, and system tasks.

Features:
    - Multi-provider support (OpenAI, Anthropic, Gemini, Ollama, custom)
    - Encrypted API key storage (Fernet symmetric encryption, key never in plaintext)
    - Model routing — assign models to task categories, with fallback chains
    - Per-model health monitoring and usage tracking
    - Hot-reload configuration changes without restart

Security Model:
    - Encryption key derived from a user-supplied master password or auto-generated
    - Keys encrypted at rest in the config file / database
    - Keys decrypted only in-memory, only when needed for API calls
    - No plaintext keys in logs, errors, or API responses
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .adapters import (
    BaseLLMAdapter,
    GeminiAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    get_adapter,
)

logger = logging.getLogger("agentvault.model_registry")


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _get_fernet():
    """Lazy import of Fernet to keep cryptography optional."""
    from cryptography.fernet import Fernet
    return Fernet


def _generate_encryption_key() -> bytes:
    """Generate a new Fernet encryption key."""
    Fernet = _get_fernet()
    return Fernet.generate_key()


def _encrypt_value(plaintext: str, key: bytes) -> str:
    """Encrypt a string and return it as a URL-safe base64 string."""
    Fernet = _get_fernet()
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def _decrypt_value(ciphertext: str, key: bytes) -> str:
    """Decrypt a URL-safe base64 string back to plaintext."""
    Fernet = _get_fernet()
    f = Fernet(key)
    return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------

class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class TaskCategory(str, Enum):
    """Categories of work that can be routed to specific models."""
    PIPELINE_DESIGN = "pipeline_design"      # Designing workflows from descriptions
    CODE_GENERATION = "code_generation"      # Writing / editing code
    DATA_ANALYSIS = "data_analysis"          # Analyzing data, SQL, etc.
    CONVERSATION = "conversation"            # Chat, customer support
    SECURITY = "security"                    # Security analysis, threat assessment
    SUMMARIZATION = "summarization"          # Summarising documents, logs
    GENERAL = "general"                      # Default / catch-all
    FAST = "fast"                            # Low-latency, cheap tasks


class ModelConfig(BaseModel):
    """Configuration for a single model provider."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                                    # Human-readable name, e.g. "GPT-4o"
    provider: ModelProvider                      # openai, anthropic, gemini, ollama
    model_id: str                                # e.g. "gpt-4o", "gemini-2.0-flash"
    api_key_encrypted: str = ""                  # Fernet-encrypted API key
    base_url: Optional[str] = None               # Custom endpoint (for OpenAI-compatible)
    is_default: bool = False                     # Default model for general tasks
    enabled: bool = True                         # Can be disabled without deletion
    max_tokens: int = 4096                       # Default max tokens
    temperature: float = 0.7                     # Default temperature

    # Routing
    task_categories: list[TaskCategory] = Field(
        default_factory=lambda: [TaskCategory.GENERAL]
    )
    priority: int = 0                            # Higher = preferred for matching categories

    # Tracking
    total_calls: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    last_used: Optional[str] = None
    last_health_check: Optional[str] = None
    is_healthy: bool = True
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ModelRegistryConfig(BaseModel):
    """Persisted registry configuration."""
    models: list[ModelConfig] = Field(default_factory=list)
    fallback_chain: list[str] = Field(default_factory=list)  # model IDs in fallback order
    version: str = "1.0"


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Central registry for all configured AI models.

    Handles:
    - Adding, updating, removing model configurations
    - Encrypting API keys at rest
    - Routing tasks to the best available model
    - Fallback chains when a model is unavailable
    - Usage tracking and health monitoring
    - Persisting config to disk (encrypted)

    Usage:
        registry = ModelRegistry()
        registry.add_model(
            name="GPT-4o",
            provider="openai",
            model_id="gpt-4o",
            api_key="sk-...",
        )
        adapter = registry.get_adapter_for_task("pipeline_design")
        result = await adapter.generate(messages=[...])
    """

    def __init__(self, config_dir: str = "config") -> None:
        self._config_dir = Path(config_dir)
        self._config_file = self._config_dir / "models.json"
        self._key_file = self._config_dir / ".encryption_key"
        self._encryption_key: Optional[bytes] = None
        self._config = ModelRegistryConfig()
        self._adapters: dict[str, BaseLLMAdapter] = {}  # cache of live adapters

    # ── Encryption key management ───────────────────────────────────

    def _ensure_encryption_key(self) -> bytes:
        """Load or create the encryption key."""
        if self._encryption_key:
            return self._encryption_key

        if self._key_file.exists():
            self._encryption_key = self._key_file.read_bytes()
        else:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            self._encryption_key = _generate_encryption_key()
            self._key_file.write_bytes(self._encryption_key)
            # Make the key file readable only by the owner
            try:
                os.chmod(str(self._key_file), 0o600)
            except (OSError, PermissionError):
                pass  # Windows may not support Unix permissions
            logger.info("Generated new encryption key at %s", self._key_file)

        return self._encryption_key

    def _encrypt_key(self, api_key: str) -> str:
        """Encrypt an API key for storage."""
        if not api_key:
            return ""
        enc_key = self._ensure_encryption_key()
        return _encrypt_value(api_key, enc_key)

    def _decrypt_key(self, encrypted: str) -> str:
        """Decrypt an API key from storage."""
        if not encrypted:
            return ""
        enc_key = self._ensure_encryption_key()
        try:
            return _decrypt_value(encrypted, enc_key)
        except Exception as e:
            logger.error("Failed to decrypt API key: %s", e)
            return ""

    # ── Persistence ─────────────────────────────────────────────────

    def save(self) -> None:
        """Save registry config to disk."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = self._config.model_dump(mode="json")
        self._config_file.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        logger.info("Saved model registry (%d models)", len(self._config.models))

    def load(self) -> None:
        """Load registry config from disk."""
        if not self._config_file.exists():
            logger.info("No model registry config found; starting fresh")
            return
        try:
            data = json.loads(self._config_file.read_text(encoding="utf-8"))
            self._config = ModelRegistryConfig(**data)
            self._adapters.clear()
            logger.info(
                "Loaded model registry: %d models", len(self._config.models)
            )
        except Exception as e:
            logger.error("Failed to load model registry: %s", e)

    # ── CRUD ────────────────────────────────────────────────────────

    def add_model(
        self,
        name: str,
        provider: str,
        model_id: str,
        api_key: str = "",
        base_url: Optional[str] = None,
        is_default: bool = False,
        task_categories: Optional[list[str]] = None,
        priority: int = 0,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ModelConfig:
        """
        Add a new model to the registry.

        Args:
            name: Human-readable name (e.g. "GPT-4o for pipelines")
            provider: openai, anthropic, gemini, ollama, custom
            model_id: Model identifier (e.g. "gpt-4o", "gemini-2.0-flash")
            api_key: Raw API key (will be encrypted before storage)
            base_url: Custom endpoint URL
            is_default: Set as default model
            task_categories: Which task types this model handles
            priority: Higher = preferred

        Returns:
            The created ModelConfig
        """
        # Encrypt the API key
        encrypted = self._encrypt_key(api_key) if api_key else ""

        # Parse provider
        try:
            prov = ModelProvider(provider.lower())
        except ValueError:
            prov = ModelProvider.CUSTOM

        # Parse categories
        categories = []
        for cat_str in (task_categories or ["general"]):
            try:
                categories.append(TaskCategory(cat_str.lower()))
            except ValueError:
                categories.append(TaskCategory.GENERAL)

        # If this is the default, unset others
        if is_default:
            for m in self._config.models:
                m.is_default = False

        config = ModelConfig(
            name=name,
            provider=prov,
            model_id=model_id,
            api_key_encrypted=encrypted,
            base_url=base_url,
            is_default=is_default,
            task_categories=categories,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self._config.models.append(config)
        self.save()
        logger.info("Added model: %s (%s/%s)", name, provider, model_id)
        return config

    def update_model(self, model_id: str, **updates) -> Optional[ModelConfig]:
        """
        Update specific fields of a model config.

        Special handling:
        - 'api_key' → encrypted before storage, stored as 'api_key_encrypted'
        """
        model = self.get_model(model_id)
        if not model:
            return None

        for key, value in updates.items():
            if key == "api_key":
                model.api_key_encrypted = self._encrypt_key(value)
            elif key == "provider":
                try:
                    model.provider = ModelProvider(value.lower())
                except ValueError:
                    model.provider = ModelProvider.CUSTOM
            elif key == "task_categories":
                cats = []
                for c in value:
                    try:
                        cats.append(TaskCategory(c.lower()))
                    except ValueError:
                        cats.append(TaskCategory.GENERAL)
                model.task_categories = cats
            elif key == "is_default" and value:
                for m in self._config.models:
                    m.is_default = False
                model.is_default = True
            elif hasattr(model, key):
                setattr(model, key, value)

        # Invalidate cached adapter
        self._adapters.pop(model.id, None)
        self.save()
        return model

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        before = len(self._config.models)
        self._config.models = [
            m for m in self._config.models if m.id != model_id
        ]
        self._adapters.pop(model_id, None)
        if len(self._config.models) < before:
            self.save()
            return True
        return False

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model config by ID."""
        for m in self._config.models:
            if m.id == model_id:
                return m
        return None

    def list_models(self, include_disabled: bool = False) -> list[dict]:
        """
        List all models (with API keys redacted).

        Returns dicts safe for API responses — keys are NEVER exposed.
        """
        results = []
        for m in self._config.models:
            if not include_disabled and not m.enabled:
                continue
            d = m.model_dump(mode="json")
            # Redact encrypted key — show only presence indicator
            d["has_api_key"] = bool(m.api_key_encrypted)
            d.pop("api_key_encrypted", None)
            results.append(d)
        return results

    def get_default_model(self) -> Optional[ModelConfig]:
        """Get the default model."""
        for m in self._config.models:
            if m.is_default and m.enabled:
                return m
        # Fallback: first enabled model
        for m in self._config.models:
            if m.enabled:
                return m
        return None

    # ── Adapter creation & routing ──────────────────────────────────

    def get_adapter(self, model_id: str) -> Optional[BaseLLMAdapter]:
        """Create or retrieve a cached adapter for a specific model."""
        if model_id in self._adapters:
            return self._adapters[model_id]

        model = self.get_model(model_id)
        if not model or not model.enabled:
            return None

        api_key = self._decrypt_key(model.api_key_encrypted)

        try:
            kwargs: dict[str, Any] = {}
            if model.base_url:
                kwargs["base_url"] = model.base_url

            adapter = get_adapter(
                provider=model.provider.value,
                api_key=api_key if api_key else None,
                model=model.model_id,
                **kwargs,
            )
            self._adapters[model_id] = adapter
            return adapter
        except Exception as e:
            logger.error("Failed to create adapter for %s: %s", model.name, e)
            return None

    def get_adapter_for_task(
        self, task_category: str = "general"
    ) -> Optional[BaseLLMAdapter]:
        """
        Get the best adapter for a task category.

        Routing logic:
        1. Find models assigned to this category
        2. Sort by priority (highest first)
        3. Filter to healthy + enabled
        4. Return the best one
        5. Fall back to default model if none match
        """
        try:
            category = TaskCategory(task_category.lower())
        except ValueError:
            category = TaskCategory.GENERAL

        # Find matching models
        candidates = [
            m for m in self._config.models
            if m.enabled and m.is_healthy and category in m.task_categories
        ]

        # Sort by priority (descending)
        candidates.sort(key=lambda m: m.priority, reverse=True)

        # Try each candidate
        for model in candidates:
            adapter = self.get_adapter(model.id)
            if adapter:
                return adapter

        # Fallback to default
        default = self.get_default_model()
        if default:
            return self.get_adapter(default.id)

        return None

    def get_all_adapters(self) -> dict[str, BaseLLMAdapter]:
        """Get adapters for all enabled models."""
        result = {}
        for m in self._config.models:
            if m.enabled:
                adapter = self.get_adapter(m.id)
                if adapter:
                    result[m.id] = adapter
        return result

    # ── Fallback chain ──────────────────────────────────────────────

    def set_fallback_chain(self, model_ids: list[str]) -> None:
        """Set the fallback chain — order of models to try if primary fails."""
        self._config.fallback_chain = model_ids
        self.save()

    def get_fallback_adapter(
        self, exclude_id: Optional[str] = None,
    ) -> Optional[BaseLLMAdapter]:
        """Get next available adapter from the fallback chain."""
        for mid in self._config.fallback_chain:
            if mid == exclude_id:
                continue
            model = self.get_model(mid)
            if model and model.enabled and model.is_healthy:
                adapter = self.get_adapter(mid)
                if adapter:
                    return adapter
        return None

    # ── Usage tracking ──────────────────────────────────────────────

    def record_usage(
        self,
        model_id: str,
        tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record usage stats for a model."""
        model = self.get_model(model_id)
        if not model:
            return

        model.total_calls += 1
        model.total_tokens += tokens

        if not success:
            model.total_errors += 1

        # Running average latency
        if model.total_calls == 1:
            model.avg_latency_ms = latency_ms
        else:
            model.avg_latency_ms = (
                (model.avg_latency_ms * (model.total_calls - 1) + latency_ms)
                / model.total_calls
            )

        model.last_used = datetime.utcnow().isoformat()

        # Auto-save periodically (every 50 calls)
        if model.total_calls % 50 == 0:
            self.save()

    # ── Health checks ───────────────────────────────────────────────

    async def check_health(self, model_id: str) -> dict:
        """
        Run a health check on a specific model.

        Returns: {"healthy": bool, "latency_ms": float, "error": str?}
        """
        adapter = self.get_adapter(model_id)
        model = self.get_model(model_id)
        if not adapter or not model:
            return {"healthy": False, "error": "Model not found or disabled"}

        start = time.monotonic()
        try:
            healthy = await adapter.health_check()
            latency = (time.monotonic() - start) * 1000

            model.is_healthy = healthy
            model.last_health_check = datetime.utcnow().isoformat()

            return {
                "healthy": healthy,
                "latency_ms": round(latency, 1),
                "model": model.name,
                "provider": model.provider.value,
            }
        except Exception as e:
            model.is_healthy = False
            model.last_health_check = datetime.utcnow().isoformat()
            return {
                "healthy": False,
                "error": str(e),
                "model": model.name,
            }

    async def check_all_health(self) -> list[dict]:
        """Run health checks on all enabled models."""
        results = []
        for m in self._config.models:
            if m.enabled:
                result = await self.check_health(m.id)
                result["id"] = m.id
                results.append(result)
        self.save()
        return results

    # ── Summary / Stats ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get registry-wide statistics."""
        models = self._config.models
        return {
            "total_models": len(models),
            "enabled": sum(1 for m in models if m.enabled),
            "healthy": sum(1 for m in models if m.is_healthy and m.enabled),
            "providers": list(set(m.provider.value for m in models)),
            "total_calls": sum(m.total_calls for m in models),
            "total_tokens": sum(m.total_tokens for m in models),
            "has_default": any(m.is_default for m in models),
            "fallback_chain_length": len(self._config.fallback_chain),
        }

    # ── Quick setup helpers ─────────────────────────────────────────

    def quick_setup_openai(self, api_key: str, model: str = "gpt-4o") -> ModelConfig:
        """Quick setup: add OpenAI with sensible defaults."""
        return self.add_model(
            name=f"OpenAI {model}",
            provider="openai",
            model_id=model,
            api_key=api_key,
            task_categories=["general", "code_generation", "pipeline_design"],
            priority=10,
        )

    def quick_setup_anthropic(
        self, api_key: str, model: str = "claude-sonnet-4-20250514"
    ) -> ModelConfig:
        """Quick setup: add Anthropic Claude with sensible defaults."""
        return self.add_model(
            name=f"Claude {model.split('-')[1] if '-' in model else model}",
            provider="anthropic",
            model_id=model,
            api_key=api_key,
            task_categories=["general", "code_generation", "conversation"],
            priority=10,
        )

    def quick_setup_gemini(
        self, api_key: str, model: str = "gemini-2.0-flash"
    ) -> ModelConfig:
        """Quick setup: add Google Gemini with sensible defaults."""
        return self.add_model(
            name=f"Gemini {model}",
            provider="gemini",
            model_id=model,
            api_key=api_key,
            task_categories=["general", "data_analysis", "fast"],
            priority=8,
        )

    def quick_setup_ollama(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ) -> ModelConfig:
        """Quick setup: add local Ollama model."""
        return self.add_model(
            name=f"Ollama {model}",
            provider="ollama",
            model_id=model,
            base_url=base_url,
            task_categories=["general", "fast", "security"],
            priority=5,
        )

    # ── Clear ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all models (for testing)."""
        self._config = ModelRegistryConfig()
        self._adapters.clear()
        self.save()
