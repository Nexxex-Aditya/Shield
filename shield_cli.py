#!/usr/bin/env python3
"""
Shield Command — CLI

Usage:
    shield compile "Monitor GitHub and notify Slack"   # Generate pipeline from description
    shield run pipeline.yaml                           # Run a pipeline file
    shield run <pipeline-id>                           # Run a saved pipeline
    shield templates                                   # List built-in templates
    shield models                                      # List configured models
    shield models add openai <api-key>                 # Quick-setup a model
    shield health                                      # Health check all models
    shield server                                      # Start the API server
    shield init [dir]                                  # Scaffold a new Shield project
    shield deploy                                      # Build & start Docker containers
    shield monitor                                     # Live monitoring dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import yaml

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _print_json(data: dict, indent: int = 2) -> None:
    print(json.dumps(data, indent=indent, default=str))


def _print_table(headers: list[str], rows: list[list]) -> None:
    """Simple table printer."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Header
    header = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header)
    print("-+-".join("-" * w for w in widths))
    
    # Rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(line)


# ---- Commands ----

def cmd_compile(args):
    """Compile a natural language description into a pipeline."""
    from agentvault.pipeline import PipelineCompiler, PipelineStore
    from agentvault.model_registry import ModelRegistry

    config_dir = os.environ.get("AGENTVAULT_CONFIG", "config")
    registry = ModelRegistry(config_dir=config_dir)
    registry.load()

    compiler = PipelineCompiler(model_registry=registry)
    store = PipelineStore(
        pipelines_dir=os.environ.get("AGENTVAULT_PIPELINES", "pipelines")
    )

    description = " ".join(args.description)
    print(f"\n🔧 Compiling pipeline: \"{description}\"\n")

    spec = asyncio.get_event_loop().run_until_complete(
        compiler.compile(description)
    )

    # Save
    filepath = store.save(spec)
    
    print(f"✅ Pipeline: {spec.name}")
    print(f"   ID: {spec.id}")
    print(f"   Steps: {len(spec.steps)}")
    print(f"   Saved: {filepath}")
    print()

    for i, step in enumerate(spec.steps, 1):
        deps = f" (after: {', '.join(step.depends_on)})" if step.depends_on else ""
        print(f"   {i}. [{step.type.value}] {step.name}{deps}")
        if step.tool_name:
            print(f"      Tool: {step.tool_name}")
        if step.prompt_template:
            preview = step.prompt_template[:80] + ("..." if len(step.prompt_template) > 80 else "")
            print(f"      Prompt: {preview}")

    if args.yaml:
        print(f"\n--- Pipeline YAML ---")
        print(yaml.dump(spec.model_dump(mode="json"), default_flow_style=False))


def cmd_run(args):
    """Run a pipeline."""
    from agentvault.pipeline import PipelineRunner, PipelineStore, PipelineSpec
    from agentvault.model_registry import ModelRegistry
    from agentvault.mcp_gateway import MCPGateway
    from agentvault.policy import PolicyEngine
    from agentvault.audit import AuditChain

    config_dir = os.environ.get("AGENTVAULT_CONFIG", "config")
    registry = ModelRegistry(config_dir=config_dir)
    registry.load()

    store = PipelineStore(
        pipelines_dir=os.environ.get("AGENTVAULT_PIPELINES", "pipelines")
    )
    store.load_all()

    target = args.target

    # Load pipeline
    pipeline = None
    if target.endswith(".yaml") or target.endswith(".yml"):
        # Load from file
        if os.path.exists(target):
            with open(target) as f:
                data = yaml.safe_load(f)
            pipeline = PipelineSpec(**data)
        else:
            print(f"❌ File not found: {target}")
            return
    else:
        # Load by ID
        pipeline = store.get(target)
        if not pipeline:
            print(f"❌ Pipeline not found: {target}")
            print("   Use 'shield templates' to see available templates")
            return

    # Set up minimal gateway
    policy = PolicyEngine()
    policy_path = os.environ.get("AGENTVAULT_POLICY", "policies/default.yaml")
    if os.path.exists(policy_path):
        policy.load(policy_path)

    audit = AuditChain()

    gateway = MCPGateway(
        policy_engine=policy,
        audit_chain=audit,
    )

    runner = PipelineRunner(gateway=gateway, model_registry=registry)

    # Parse context
    context = {}
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            # Try key=value pairs
            for pair in args.context.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    context[k.strip()] = v.strip()

    print(f"\n🚀 Running pipeline: {pipeline.name}")
    print(f"   Steps: {len(pipeline.steps)}")
    if context:
        print(f"   Context: {context}")
    print()

    result = asyncio.get_event_loop().run_until_complete(
        runner.run(pipeline, initial_context=context)
    )

    # Print results
    status_emoji = "✅" if result.status.value == "completed" else "❌"
    print(f"\n{status_emoji} Pipeline {result.status.value}")
    print(f"   Completed: {result.steps_completed}/{result.total_steps}")
    print(f"   Failed: {result.steps_failed}")
    print(f"   Duration: {result.duration_ms:.0f}ms")

    if args.verbose:
        print("\n--- Step Results ---")
        for sr in result.step_results:
            s_emoji = "✅" if sr["status"] == "completed" else "❌"
            print(f"   {s_emoji} {sr['step_name']}: {sr['status']} ({sr['duration_ms']:.0f}ms)")
            if sr.get("error"):
                print(f"      Error: {sr['error']}")


def cmd_templates(args):
    """List built-in pipeline templates."""
    from agentvault.pipeline import PipelineCompiler

    compiler = PipelineCompiler()
    templates = compiler.list_templates()

    print(f"\n📋 Built-in Pipeline Templates ({len(templates)})\n")
    _print_table(
        ["Name", "Steps", "Tags"],
        [[t["name"], t["steps"], ", ".join(t["tags"][:4])] for t in templates],
    )
    print()


def cmd_models(args):
    """List or manage configured models."""
    from agentvault.model_registry import ModelRegistry

    config_dir = os.environ.get("AGENTVAULT_CONFIG", "config")
    registry = ModelRegistry(config_dir=config_dir)
    registry.load()

    if args.action == "list" or args.action is None:
        models = registry.list_models(include_disabled=True)
        if not models:
            print("\n⚠️  No models configured. Use 'shield models add <provider> <api-key>'")
            return

        print(f"\n🤖 Configured Models ({len(models)})\n")
        _print_table(
            ["ID", "Name", "Provider", "Model", "Default", "Key", "Enabled"],
            [
                [
                    m["id"], m["name"], m["provider"], m["model_id"],
                    "★" if m["is_default"] else "", "✓" if m["has_api_key"] else "✗",
                    "✓" if m["enabled"] else "✗",
                ]
                for m in models
            ],
        )
        print()

    elif args.action == "add":
        if not args.provider:
            print("Usage: shield models add <provider> [api-key] [--model MODEL]")
            return
        provider = args.provider
        api_key = args.api_key or ""
        model = getattr(args, "model_name", None)

        if provider == "openai":
            m = registry.quick_setup_openai(api_key, model=model or "gpt-4o")
        elif provider in ("anthropic", "claude"):
            m = registry.quick_setup_anthropic(api_key, model=model or "claude-sonnet-4-20250514")
        elif provider in ("gemini", "google"):
            m = registry.quick_setup_gemini(api_key, model=model or "gemini-2.0-flash")
        elif provider == "ollama":
            m = registry.quick_setup_ollama(model=model or "llama3.2")
        else:
            print(f"❌ Unknown provider: {provider}")
            return
        print(f"✅ Added: {m.name} (ID: {m.id})")

    elif args.action == "remove":
        if not args.provider:
            print("Usage: shield models remove <model-id>")
            return
        removed = registry.remove_model(args.provider)
        if removed:
            print(f"✅ Removed model: {args.provider}")
        else:
            print(f"❌ Model not found: {args.provider}")


def cmd_health(args):
    """Health check all configured models."""
    from agentvault.model_registry import ModelRegistry

    config_dir = os.environ.get("AGENTVAULT_CONFIG", "config")
    registry = ModelRegistry(config_dir=config_dir)
    registry.load()

    models = registry.list_models()
    if not models:
        print("⚠️  No models configured")
        return

    print(f"\n🏥 Health Check ({len(models)} models)\n")

    results = asyncio.get_event_loop().run_until_complete(
        registry.check_all_health()
    )

    for r in results:
        emoji = "✅" if r.get("healthy") else "❌"
        name = r.get("model", r.get("id", "?"))
        latency = f" ({r['latency_ms']:.0f}ms)" if "latency_ms" in r else ""
        error = f" — {r['error']}" if "error" in r else ""
        print(f"   {emoji} {name}{latency}{error}")
    print()


def cmd_server(args):
    """Start the Shield API server."""
    import uvicorn
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    print(f"\n🛡️  Starting Shield Command Server at http://{host}:{port}")
    print(f"   Dashboard: http://localhost:{port}")
    print(f"   API docs:  http://localhost:{port}/docs\n")
    uvicorn.run("server.app:app", host=host, port=port, reload=args.reload)


def cmd_init(args):
    """Scaffold a new Shield project."""
    target = args.directory or "."
    if target != ".":
        os.makedirs(target, exist_ok=True)
    
    dirs = ["pipelines", "policies", "skills", "config", "data"]
    for d in dirs:
        os.makedirs(os.path.join(target, d), exist_ok=True)
    
    # Default policy
    policy_path = os.path.join(target, "policies", "default.yaml")
    if not os.path.exists(policy_path):
        with open(policy_path, "w") as f:
            yaml.dump({
                "version": "1.0",
                "rules": [
                    {"pattern": "*", "decision": "allow", "description": "Allow all tools (edit for production)"},
                ],
            }, f, default_flow_style=False)
    
    # .env template
    env_path = os.path.join(target, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("""# Shield Configuration
DB_URI=sqlite+aiosqlite:///data/shield.db
POLICY_PATH=policies/default.yaml
PIPELINES_DIR=pipelines
SKILLS_DIR=skills

# API Keys (uncomment and set yours)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=AIza...
""")
    
    print(f"""\n🛡️  Shield project initialized in: {os.path.abspath(target)}

   📁 pipelines/     → Your pipeline YAML files
   📁 policies/      → Security policy rules
   📁 skills/        → Skill definitions
   📁 config/        → Model registry configs
   📁 data/          → Database & runtime data
   📄 .env           → Environment variables

   Next steps:
     1. Edit .env and add your API keys
     2. shield models add openai <key>   # Configure a model
     3. shield server                    # Start the server
     4. Open http://localhost:8000       # Dashboard
""")


def cmd_deploy(args):
    """Build and deploy Shield with Docker."""
    compose_file = "docker-compose.yml"
    
    if not os.path.exists(compose_file):
        print("❌ docker-compose.yml not found. Run from the Shield project root.")
        return
    
    print("\n🐳 Shield Docker Deployment\n")
    
    if args.build:
        print("   Building image...")
        result = subprocess.run(
            ["docker-compose", "build"],
            capture_output=not args.verbose,
        )
        if result.returncode != 0:
            print("❌ Build failed")
            if not args.verbose and result.stderr:
                print(result.stderr.decode()[:500])
            return
        print("   ✅ Image built\n")
    
    action = "up -d" if args.detach else "up"
    print(f"   Starting containers ({action})...")
    cmd = ["docker-compose"] + action.split()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n   Shutting down...")
        subprocess.run(["docker-compose", "down"])
    
    if args.detach:
        print("""\n   ✅ Shield is running!
     Dashboard: http://localhost:8000
     API docs:  http://localhost:8000/docs
     Logs:      docker-compose logs -f shield
     Stop:      docker-compose down
""")


def cmd_monitor(args):
    """Live monitoring dashboard in terminal."""
    import urllib.request
    
    base_url = args.url or "http://localhost:8000"
    interval = args.interval or 5
    
    print(f"\n📡 Shield Monitor — {base_url} (refresh: {interval}s)")
    print("   Press Ctrl+C to exit\n")
    
    api = f"{base_url}/api/v1"
    
    def fetch(path):
        try:
            req = urllib.request.Request(f"{api}{path}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:
            return None
    
    try:
        while True:
            # Clear screen
            os.system("cls" if os.name == "nt" else "clear")
            now = time.strftime("%H:%M:%S")
            print(f"🛡️  Shield Monitor  [{now}]  ({base_url})")
            print("=" * 60)
            
            # Models
            models = fetch("/models")
            if models and models.get("models"):
                mlist = models["models"]
                enabled = sum(1 for m in mlist if m.get("enabled", True))
                print(f"\n🤖 Models: {len(mlist)} total, {enabled} enabled")
                for m in mlist[:5]:
                    status = "✅" if m.get("enabled", True) else "⏸️"
                    print(f"   {status} {m.get('name', m.get('model_id', '?'))} ({m.get('provider', '?')})")
            else:
                print("\n🤖 Models: unavailable")
            
            # Pipelines
            pipes = fetch("/pipelines")
            if pipes and pipes.get("pipelines"):
                plist = pipes["pipelines"]
                print(f"\n🔧 Pipelines: {len(plist)}")
                for p in plist[:5]:
                    runs = p.get("run_count", 0)
                    status = p.get("status", "draft")
                    print(f"   • {p.get('name', p.get('id', '?'))} — {runs} runs [{status}]")
            else:
                print("\n🔧 Pipelines: none")
            
            # Pipeline stats
            stats = fetch("/pipelines/stats")
            if stats and stats.get("totals"):
                t = stats["totals"]
                print(f"\n📊 Pipeline Stats:")
                print(f"   Runs: {t.get('total_runs', 0)}  Success: {t.get('success_rate', 0)}%  Avg: {t.get('avg_duration_ms', 0):.0f}ms")
            
            # Health
            health = fetch("/health")
            if health:
                print(f"\n💚 Server: healthy")
            else:
                print(f"\n🔴 Server: unreachable")
            
            print(f"\n{'─' * 60}")
            print(f"  Refresh in {interval}s...  [Ctrl+C to exit]")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")



# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        prog="shield",
        description="Shield Command — Autonomous Operations Platform",
    )
    subparsers = parser.add_subparsers(dest="command")

    # compile
    compile_p = subparsers.add_parser("compile", help="Compile description → pipeline")
    compile_p.add_argument("description", nargs="+", help="Pipeline description")
    compile_p.add_argument("--yaml", action="store_true", help="Print YAML output")

    # run
    run_p = subparsers.add_parser("run", help="Run a pipeline")
    run_p.add_argument("target", help="Pipeline ID or YAML file path")
    run_p.add_argument("--context", "-c", help="Context as JSON or key=value pairs")
    run_p.add_argument("--verbose", "-v", action="store_true")

    # templates
    subparsers.add_parser("templates", help="List built-in templates")

    # models
    models_p = subparsers.add_parser("models", help="Manage AI models")
    models_p.add_argument("action", nargs="?", choices=["list", "add", "remove"])
    models_p.add_argument("provider", nargs="?")
    models_p.add_argument("api_key", nargs="?")
    models_p.add_argument("--model-name", help="Model ID override")

    # health
    subparsers.add_parser("health", help="Health check all models")

    # server
    server_p = subparsers.add_parser("server", help="Start API server")
    server_p.add_argument("--host", default="0.0.0.0")
    server_p.add_argument("--port", type=int, default=8000)
    server_p.add_argument("--reload", action="store_true")

    # init
    init_p = subparsers.add_parser("init", help="Scaffold a new Shield project")
    init_p.add_argument("directory", nargs="?", default=".", help="Target directory")

    # deploy
    deploy_p = subparsers.add_parser("deploy", help="Build & start Docker containers")
    deploy_p.add_argument("--build", "-b", action="store_true", help="Build image first")
    deploy_p.add_argument("--detach", "-d", action="store_true", help="Run in background")
    deploy_p.add_argument("--verbose", "-v", action="store_true")

    # monitor
    monitor_p = subparsers.add_parser("monitor", help="Live terminal monitoring")
    monitor_p.add_argument("--url", default="http://localhost:8000", help="Server URL")
    monitor_p.add_argument("--interval", "-i", type=int, default=5, help="Refresh interval")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "compile": cmd_compile,
        "run": cmd_run,
        "templates": cmd_templates,
        "models": cmd_models,
        "health": cmd_health,
        "server": cmd_server,
        "init": cmd_init,
        "deploy": cmd_deploy,
        "monitor": cmd_monitor,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
