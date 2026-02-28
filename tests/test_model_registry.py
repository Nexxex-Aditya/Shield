"""
Tests for ModelRegistry — model configuration, encrypted key storage, 
routing, and factory integration.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.model_registry import (
    ModelRegistry,
    ModelConfig,
    ModelProvider,
    TaskCategory,
    _encrypt_value,
    _decrypt_value,
    _generate_encryption_key,
)


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config directory."""
    return str(tmp_path / "test_config")


@pytest.fixture
def registry(tmp_config):
    """Create a fresh ModelRegistry instance."""
    r = ModelRegistry(config_dir=tmp_config)
    return r


# -------------------------------------------------------------------------
# Encryption
# -------------------------------------------------------------------------

class TestEncryption:
    def test_roundtrip(self):
        key = _generate_encryption_key()
        original = "sk-test-super-secret-key-12345"
        encrypted = _encrypt_value(original, key)
        assert encrypted != original
        decrypted = _decrypt_value(encrypted, key)
        assert decrypted == original

    def test_different_keys_fail(self):
        key1 = _generate_encryption_key()
        key2 = _generate_encryption_key()
        encrypted = _encrypt_value("secret", key1)
        with pytest.raises(Exception):
            _decrypt_value(encrypted, key2)

    def test_empty_string(self):
        key = _generate_encryption_key()
        encrypted = _encrypt_value("", key)
        decrypted = _decrypt_value(encrypted, key)
        assert decrypted == ""


# -------------------------------------------------------------------------
# CRUD
# -------------------------------------------------------------------------

class TestModelCRUD:
    def test_add_model(self, registry):
        model = registry.add_model(
            name="Test GPT",
            provider="openai",
            model_id="gpt-4o",
            api_key="sk-test-123",
        )
        assert model.name == "Test GPT"
        assert model.provider == ModelProvider.OPENAI
        assert model.model_id == "gpt-4o"
        assert model.api_key_encrypted != ""
        assert model.api_key_encrypted != "sk-test-123"  # encrypted, not plain

    def test_list_models_redacted(self, registry):
        registry.add_model(
            name="Test", provider="openai", model_id="gpt-4o",
            api_key="sk-secret",
        )
        models = registry.list_models()
        assert len(models) == 1
        assert "api_key_encrypted" not in models[0]
        assert models[0]["has_api_key"] is True

    def test_list_disabled(self, registry):
        m = registry.add_model(name="D", provider="openai", model_id="gpt-4o")
        registry.update_model(m.id, enabled=False)
        assert len(registry.list_models()) == 0
        assert len(registry.list_models(include_disabled=True)) == 1

    def test_get_model(self, registry):
        m = registry.add_model(name="T", provider="gemini", model_id="gemini-2.0-flash")
        found = registry.get_model(m.id)
        assert found is not None
        assert found.name == "T"

    def test_get_missing(self, registry):
        assert registry.get_model("nonexistent") is None

    def test_update_model(self, registry):
        m = registry.add_model(name="Old", provider="openai", model_id="gpt-4o")
        updated = registry.update_model(m.id, name="New")
        assert updated.name == "New"

    def test_update_api_key(self, registry):
        m = registry.add_model(
            name="T", provider="openai", model_id="gpt-4o", api_key="old-key"
        )
        old_enc = m.api_key_encrypted
        registry.update_model(m.id, api_key="new-key")
        assert m.api_key_encrypted != old_enc

    def test_remove_model(self, registry):
        m = registry.add_model(name="R", provider="openai", model_id="gpt-4o")
        assert registry.remove_model(m.id) is True
        assert registry.get_model(m.id) is None

    def test_remove_nonexistent(self, registry):
        assert registry.remove_model("xxx") is False

    def test_default_model(self, registry):
        registry.add_model(name="A", provider="openai", model_id="gpt-4o")
        registry.add_model(
            name="B", provider="gemini", model_id="gemini-2.0-flash", is_default=True
        )
        default = registry.get_default_model()
        assert default.name == "B"

    def test_only_one_default(self, registry):
        m1 = registry.add_model(name="A", provider="openai", model_id="x", is_default=True)
        m2 = registry.add_model(name="B", provider="gemini", model_id="y", is_default=True)
        assert m1.is_default is False
        assert m2.is_default is True


# -------------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_config):
        r1 = ModelRegistry(config_dir=tmp_config)
        r1.add_model(name="Saved", provider="openai", model_id="gpt-4o", api_key="secret")

        r2 = ModelRegistry(config_dir=tmp_config)
        r2.load()
        models = r2.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "Saved"
        assert models[0]["has_api_key"] is True

    def test_encryption_key_persisted(self, tmp_config):
        r1 = ModelRegistry(config_dir=tmp_config)
        r1.add_model(name="T", provider="openai", model_id="x", api_key="my-key")
        key1 = r1._encryption_key

        r2 = ModelRegistry(config_dir=tmp_config)
        r2.load()
        key2 = r2._ensure_encryption_key()
        assert key1 == key2

    def test_decryption_after_reload(self, tmp_config):
        r1 = ModelRegistry(config_dir=tmp_config)
        r1.add_model(name="T", provider="openai", model_id="x", api_key="super-secret")

        r2 = ModelRegistry(config_dir=tmp_config)
        r2.load()
        model = r2._config.models[0]
        decrypted = r2._decrypt_key(model.api_key_encrypted)
        assert decrypted == "super-secret"


# -------------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------------

class TestRouting:
    def test_route_by_category(self, registry):
        registry.add_model(
            name="CodeModel", provider="ollama", model_id="codellama",
            task_categories=["code_generation"], priority=10,
        )
        registry.add_model(
            name="ChatModel", provider="ollama", model_id="llama3.2",
            task_categories=["conversation"], priority=10,
        )
        adapter = registry.get_adapter_for_task("code_generation")
        assert adapter is not None
        assert "codellama" in adapter.name.lower()

    def test_priority_ordering(self, registry):
        registry.add_model(
            name="Low", provider="ollama", model_id="low",
            task_categories=["general"], priority=1,
        )
        registry.add_model(
            name="High", provider="ollama", model_id="high",
            task_categories=["general"], priority=100,
        )
        adapter = registry.get_adapter_for_task("general")
        assert "high" in adapter.name.lower()

    def test_fallback_to_default(self, registry):
        registry.add_model(
            name="Default", provider="ollama", model_id="default",
            task_categories=["general"], is_default=True,
        )
        # No model for "security" category, should fallback to default
        adapter = registry.get_adapter_for_task("security")
        assert adapter is not None
        assert "default" in adapter.name.lower()


# -------------------------------------------------------------------------
# Quick Setup Helpers
# -------------------------------------------------------------------------

class TestQuickSetup:
    def test_openai_setup(self, registry):
        m = registry.quick_setup_openai("sk-test")
        assert m.provider == ModelProvider.OPENAI
        assert TaskCategory.PIPELINE_DESIGN in m.task_categories

    def test_anthropic_setup(self, registry):
        m = registry.quick_setup_anthropic("sk-ant-test")
        assert m.provider == ModelProvider.ANTHROPIC
        assert TaskCategory.CONVERSATION in m.task_categories

    def test_gemini_setup(self, registry):
        m = registry.quick_setup_gemini("key")
        assert m.provider == ModelProvider.GEMINI
        assert TaskCategory.FAST in m.task_categories

    def test_ollama_setup(self, registry):
        m = registry.quick_setup_ollama()
        assert m.provider == ModelProvider.OLLAMA
        assert m.api_key_encrypted == ""


# -------------------------------------------------------------------------
# Stats
# -------------------------------------------------------------------------

class TestStats:
    def test_stats(self, registry):
        registry.add_model(name="A", provider="openai", model_id="x")
        registry.add_model(name="B", provider="gemini", model_id="y")
        stats = registry.get_stats()
        assert stats["total_models"] == 2
        assert stats["enabled"] == 2
        assert "openai" in stats["providers"]
        assert "gemini" in stats["providers"]

    def test_usage_tracking(self, registry):
        m = registry.add_model(name="T", provider="openai", model_id="x")
        registry.record_usage(m.id, tokens=100, latency_ms=50.0, success=True)
        registry.record_usage(m.id, tokens=200, latency_ms=100.0, success=False)
        model = registry.get_model(m.id)
        assert model.total_calls == 2
        assert model.total_tokens == 300
        assert model.total_errors == 1
        assert model.avg_latency_ms == 75.0


# -------------------------------------------------------------------------
# Import Check
# -------------------------------------------------------------------------

class TestImports:
    def test_from_package(self):
        from agentvault import ModelRegistry, ModelProvider, TaskCategory
        assert ModelRegistry is not None

    def test_in_all(self):
        import agentvault
        assert "ModelRegistry" in agentvault.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
