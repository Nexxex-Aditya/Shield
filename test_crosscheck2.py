"""Connectivity cross-check - ASCII safe"""
import sys, os, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

fails = []

# Test registry
try:
    from agentvault.registry import ServiceRegistry
    r = ServiceRegistry()
    r.load_builtins()
    s = r.connect("sqlite", {"database": ":memory:"}, "test_db")
    assert s.connected
    conns = r.list_connections()
    stats = r.get_stats()
    print("Registry: OK")
except Exception as e:
    fails.append("Registry: " + str(e))
    print("Registry: FAIL " + str(e))

# Test shadow
try:
    from agentvault.shadow import ShadowEngine
    s = ShadowEngine()
    assert s.should_shadow("delete_file", {"path": "/data"})
    r = s.evaluate("execute_sql", {"query": "DROP TABLE users"})
    assert r.verdict in ("escalate", "block")
    print("Shadow: OK verdict=" + r.verdict)
except Exception as e:
    fails.append("Shadow: " + str(e))
    print("Shadow: FAIL " + str(e))

# Test surveillance
try:
    from agentvault.surveillance import ResponseSurveillance
    sv = ResponseSurveillance()
    for i in range(20):
        sv.record_response("t", True, 100+(i%3), 500+i)
    a = sv.record_response("t", True, 10000.0, 500)
    health = sv.get_tool_health()
    anomaly_type = a.anomaly_type if a else "None"
    print("Surveillance: OK anomaly=" + anomaly_type)
except Exception as e:
    fails.append("Surveillance: " + str(e))
    print("Surveillance: FAIL " + str(e))

# CIBIL
try:
    from agentvault.cibil import CIBILEngine
    c = CIBILEngine()
    for i in range(30):
        c.record_action("gpt-4o", "search_web", True, confidence=0.85, latency_ms=200)
    for i in range(20):
        c.record_action("gpt-4o", "execute_code", i<16, confidence=0.7, latency_ms=500)
    for i in range(25):
        c.record_action("claude-3.5", "execute_code", i<23, confidence=0.92, latency_ms=300)
    card = c.get_report_card("gpt-4o")
    models = c.list_models()
    score = c.get_score("gpt-4o")
    print("CIBIL: OK score=" + str(round(score)) + " grade=" + card.grade + " models=" + str(len(models)))
except Exception as e:
    fails.append("CIBIL: " + str(e))
    print("CIBIL: FAIL " + str(e))

# Recommendations
try:
    from agentvault.recommendations import RecommendationEngine
    rec = RecommendationEngine(c)
    recs = rec.suggest_model("development")
    warns = rec.get_warnings("gpt-4o")
    report = rec.generate_report("gpt-4o")
    dash = rec.get_dashboard_summary()
    print("Recommendations: OK suggestions=" + str(len(recs)) + " warnings=" + str(len(warns)))
except Exception as e:
    fails.append("Recommendations: " + str(e))
    print("Recommendations: FAIL " + str(e))

# Gateway
try:
    from agentvault.mcp_gateway import MCPGateway
    from agentvault.policy import PolicyEngine
    from agentvault.audit import AuditChain
    from agentvault.drift import DriftDetector
    from agentvault.confidence import ConfidenceScorer
    gw = MCPGateway(
        policy_engine=PolicyEngine(), audit_chain=AuditChain(),
        drift_detector=DriftDetector(), confidence_scorer=ConfidenceScorer(),
        shadow_engine=ShadowEngine(), surveillance=ResponseSurveillance(),
    )
    assert hasattr(gw, "_shadow")
    assert hasattr(gw, "_surveillance")
    print("MCPGateway 12-step: OK")
except Exception as e:
    fails.append("MCPGateway: " + str(e))
    print("MCPGateway: FAIL " + str(e))

# Routes
try:
    from server.routes import set_dependencies
    sig = inspect.signature(set_dependencies)
    params = list(sig.parameters.keys())
    assert "registry" in params
    assert "surveillance" in params
    assert "shadow_engine" in params
    assert "cibil_engine" in params
    assert "recommendation_engine" in params
    print("Routes: OK params=" + str(params))
except Exception as e:
    fails.append("Routes: " + str(e))
    print("Routes: FAIL " + str(e))

# Version
import agentvault
print("Version: " + agentvault.__version__)
assert agentvault.__version__ == "0.3.0"

print("---")
print("Total failures: " + str(len(fails)))
for f in fails:
    print("  FAIL: " + f)
if not fails:
    print("ALL CONNECTED - PROJECT FULLY WIRED")
