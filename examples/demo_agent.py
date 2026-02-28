"""
AgentVault — Demo Agent
A fully functional agent demonstrating:
  - Policy enforcement (ALLOW / DENY / ESCALATE)
  - Sandbox violations (path restriction, PII detection)
  - Drift detection
  - Full audit chain

Usage:
  python examples/demo_agent.py

Pre-requisites:
  pip install -r requirements.txt  
  (optional) ollama pull llama3.2   # for LLM mode
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.policy import PolicyEngine
from agentvault.audit import AuditChain
from agentvault.drift import DriftDetector
from agentvault.confidence import ConfidenceScorer
from agentvault.sandbox import ToolSandbox, SandboxViolationError
from agentvault.mcp_gateway import MCPGateway
from agentvault.models import SandboxConfig


# ──────────────────────────────────────────────────────────────
# Mock Tools (simulated — no actual files/network)
# ──────────────────────────────────────────────────────────────

async def read_file(path: str = "") -> str:
    """Simulate reading a file."""
    fake_data = {
        "/data/sales.csv": "date,amount\n2024-01-01,15000\n2024-01-02,18000\n2024-01-03,12500",
        "/data/users.csv": "name,email\nJohn Doe,john@example.com\nJane Smith,jane@example.com",
    }
    return fake_data.get(path, f"File contents of {path}: [sample data]")


async def write_file(path: str = "", content: str = "") -> str:
    """Simulate writing a file."""
    return f"Written {len(content)} bytes to {path}"


async def list_files(directory: str = "/data") -> str:
    """Simulate listing files."""
    return json.dumps(["sales.csv", "users.csv", "config.yaml", "report.pdf"])


async def search_data(query: str = "", dataset: str = "sales") -> str:
    """Simulate searching data."""
    return json.dumps({
        "query": query,
        "results": [
            {"id": 1, "match": f"Result for '{query}' in {dataset}", "score": 0.95},
            {"id": 2, "match": f"Secondary result for '{query}'", "score": 0.82},
        ],
    })


async def analyze_sales(period: str = "monthly") -> str:
    """Simulate analyzing sales data."""
    return json.dumps({
        "period": period,
        "total_revenue": 245000,
        "growth": 12.5,
        "top_product": "Enterprise License",
    })


async def delete_file(path: str = "") -> str:
    """Simulate deleting a file (should be DENIED)."""
    return f"DELETED {path}"


async def send_email(to: str = "", subject: str = "", body: str = "") -> str:
    """Simulate sending an email (should be ESCALATED)."""
    return f"Email sent to {to}: {subject}"


async def execute_code(code: str = "") -> str:
    """Simulate code execution."""
    return f"Executed code: {code[:50]}... → result: 42"


async def api_call(url: str = "", method: str = "GET") -> str:
    """Simulate an API call."""
    return json.dumps({"status": 200, "data": {"message": "API response from " + url}})


async def read_sensitive(path: str = "/etc/passwd") -> str:
    """Return data with PII (tests PII scanner)."""
    return """
    User record:
    Name: John Smith
    SSN: 123-45-6789
    Credit Card: 4532-1234-5678-9012
    Email: john.smith@company.com
    Phone: (555) 123-4567
    API Key: sk-abc123def456ghi789jkl012mno345
    """


# ──────────────────────────────────────────────────────────────
# Demo Runner
# ──────────────────────────────────────────────────────────────

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🛡️  AgentVault — Live Demo                       ║
║          Secure MCP Gateway + Agent Firewall              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")


def section(title: str):
    print(f"\n{PURPLE}{BOLD}{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}{RESET}\n")


def result_line(tool: str, decision: str, reason: str = ""):
    icon = {
        "ALLOW": f"{GREEN}✓ ALLOW{RESET}",
        "DENY": f"{RED}✗ DENY{RESET}",
        "ESCALATE": f"{YELLOW}⚠ ESCALATE{RESET}",
    }.get(decision, decision)

    print(f"  {DIM}→{RESET} {BOLD}{tool}{RESET}  {icon}  {DIM}{reason}{RESET}")


async def main():
    banner()

    # ── Initialize ────────────────────────────────────────────
    print(f"{BOLD}Initializing AgentVault...{RESET}")

    policy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "policies", "default.yaml"
    )

    policy = PolicyEngine()
    policy.load(policy_path)
    print(f"  {GREEN}✓{RESET} Policies loaded: {len(policy.policies)} configs")

    audit = AuditChain()
    print(f"  {GREEN}✓{RESET} Audit chain initialized")

    drift = DriftDetector()
    print(f"  {GREEN}✓{RESET} Drift detector ready")

    confidence = ConfidenceScorer()
    print(f"  {GREEN}✓{RESET} Confidence scorer ready")

    sandbox_config = SandboxConfig(
        allowed_paths=["/data", "/tmp"],
        allowed_domains=["localhost", "api.example.com"],
        max_execution_time=10,
        scan_pii=True,
    )

    gateway = MCPGateway(
        policy_engine=policy,
        audit_chain=audit,
        drift_detector=drift,
        confidence_scorer=confidence,
        sandbox_config=sandbox_config,
    )

    # Register tools
    gateway.register_tool("read_file", read_file, "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}})
    gateway.register_tool("write_file", write_file, "Write a file", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}})
    gateway.register_tool("list_files", list_files, "List files in directory")
    gateway.register_tool("search_data", search_data, "Search a dataset")
    gateway.register_tool("analyze_sales", analyze_sales, "Analyze sales data")
    gateway.register_tool("delete_file", delete_file, "Delete a file")
    gateway.register_tool("send_email", send_email, "Send an email")
    gateway.register_tool("execute_code", execute_code, "Execute code")
    gateway.register_tool("api_call", api_call, "Make an API call")
    gateway.register_tool("read_sensitive", read_sensitive, "Read sensitive data")

    print(f"  {GREEN}✓{RESET} {len(gateway.list_tools())} tools registered")
    print(f"\n{GREEN}{BOLD}AgentVault ready!{RESET}\n")

    agent_id = "demo-agent"
    session_id = "demo-session-001"

    # ── SCENARIO 1: Normal Operations (ALLOW) ─────────────────
    section("SCENARIO 1: Normal Operations → ALLOW")

    allow_tests = [
        ("read_file", {"path": "/data/sales.csv"}),
        ("list_files", {"directory": "/data"}),
        ("search_data", {"query": "revenue", "dataset": "sales"}),
        ("analyze_sales", {"period": "quarterly"}),
    ]

    for tool_name, params in allow_tests:
        result = await gateway.handle_tool_call(agent_id, session_id, tool_name, params)
        result_line(tool_name, result["decision"], result.get("reasoning", ""))
        await asyncio.sleep(0.1)

    # ── SCENARIO 2: Dangerous Operations (DENY) ──────────────
    section("SCENARIO 2: Dangerous Operations → DENY")

    deny_tests = [
        ("delete_file", {"path": "/data/important.csv"}),
    ]

    for tool_name, params in deny_tests:
        result = await gateway.handle_tool_call(agent_id, session_id, tool_name, params)
        result_line(tool_name, result["decision"], result.get("reasoning", ""))

    # ── SCENARIO 3: Operations Requiring Approval (ESCALATE) ─
    section("SCENARIO 3: Human Approval Required → ESCALATE")

    escalate_tests = [
        ("send_email", {"to": "client@corp.com", "subject": "Q4 Results", "body": "Please find attached..."}),
    ]

    for tool_name, params in escalate_tests:
        result = await gateway.handle_tool_call(agent_id, session_id, tool_name, params)
        result_line(tool_name, result["decision"], result.get("reasoning", ""))

    # ── SCENARIO 4: PII Detection ────────────────────────────
    section("SCENARIO 4: PII Detection & Redaction")

    result = await gateway.handle_tool_call(agent_id, session_id, "read_sensitive", {"path": "/data/users_full.csv"})
    result_line("read_sensitive", result["decision"], result.get("reasoning", ""))

    if result.get("result") and isinstance(result["result"], str):
        print(f"\n  {YELLOW}PII scan result (redacted):{RESET}")
        for line in result["result"].strip().split("\n"):
            line = line.strip()
            if line:
                print(f"    {DIM}{line}{RESET}")

    # ── SCENARIO 5: Rate Limiting ────────────────────────────
    section("SCENARIO 5: Rate Limiting (write_file)")

    for i in range(5):
        result = await gateway.handle_tool_call(
            agent_id, session_id, "write_file",
            {"path": f"/data/output_{i}.txt", "content": f"Data batch {i}"}
        )
        result_line(f"write_file #{i+1}", result["decision"], result.get("reasoning", ""))

    # ── SCENARIO 6: Drift Detection ──────────────────────────
    section("SCENARIO 6: Behavioral Drift Detection")

    # Create a second agent with unusual patterns
    drift_agent = "rogue-agent"
    print(f"  {DIM}Simulating unusual behavior from '{drift_agent}'...{RESET}")

    # Burst of delete attempts (should trigger drift + deny)
    for i in range(8):
        result = await gateway.handle_tool_call(
            drift_agent, "drift-session",
            "delete_file", {"path": f"/critical/file_{i}.db"}
        )

    # Check for drift alerts
    alerts = drift.get_alerts()
    if alerts:
        for alert in alerts[-3:]:
            print(f"  {RED}🚨 DRIFT:{RESET} {alert.agent_id} — "
                  f"Level: {alert.alert_level.value}, "
                  f"Score: {alert.deviation_score:.1f}")
    else:
        print(f"  {DIM}No drift alerts yet (needs more baseline data){RESET}")

    # ── SCENARIO 7: Confidence Scoring ───────────────────────
    section("SCENARIO 7: Confidence Scoring")

    test_texts = [
        ("High confidence", "The quarterly revenue increased by 15% to $2.4M."),
        ("Low confidence", "I think maybe the results could possibly be around $2M, but I'm not really sure about that, although it might also be different."),
        ("Contradictory", "Revenue went up significantly. However, revenue actually decreased this quarter."),
    ]

    for label, text in test_texts:
        score = confidence.score(text)
        color = GREEN if score.value > 0.7 else YELLOW if score.value > 0.4 else RED
        print(f"  {label}: {color}{score.value:.2f}{RESET} — {DIM}{score.reasoning}{RESET}")

    # ── Audit Summary ────────────────────────────────────────
    section("AUDIT CHAIN SUMMARY")

    stats = audit.get_stats()
    valid, break_idx = audit.verify()

    print(f"  Total events:    {BOLD}{stats['total']}{RESET}")
    print(f"  Allowed:         {GREEN}{stats['allowed']}{RESET}")
    print(f"  Denied:          {RED}{stats['denied']}{RESET}")
    print(f"  Escalated:       {YELLOW}{stats['escalated']}{RESET}")
    print(f"  Chain integrity: {GREEN if valid else RED}{'✓ VALID' if valid else '✗ BROKEN'}{RESET}")

    # ── Export ────────────────────────────────────────────────
    section("EXPORT")

    export_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "demo_audit_export.json"
    )
    json_str = audit.export_json()
    with open(export_path, "w") as f:
        f.write(json_str)
    print(f"  Audit log exported to: {CYAN}{export_path}{RESET}")

    # ── Done ─────────────────────────────────────────────────
    print(f"\n{GREEN}{BOLD}╔══════════════════════════════════════════════════════════╗")
    print(f"║  Demo complete! All scenarios executed successfully.     ║")
    print(f"║                                                          ║")
    print(f"║  Next: Start the server:                                 ║")
    print(f"║    uvicorn server.app:app --reload                       ║")
    print(f"║  Then open http://localhost:8000 for the dashboard.      ║")
    print(f"╚══════════════════════════════════════════════════════════╝{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
