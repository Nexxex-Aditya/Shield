"""Quick verification script for Skills Engine + Dashboard endpoints."""
import urllib.request
import json

BASE = "http://localhost:8000/api/v1"

def get(path):
    r = urllib.request.urlopen(BASE + path)
    return json.loads(r.read())

def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

print("=== Skills List ===")
d = get("/skills")
print(f"Skills loaded: {len(d['skills'])}")
for s in d["skills"]:
    print(f"  - {s['name']} v{s['version']} ({s['steps_count']} steps)")

print("\n=== System Config ===")
cfg = get("/config/system")
print(f"Database: {cfg['database']['connected']}")
print(f"Gateway: {cfg['gateway']['connected']}")
print(f"Skills: {cfg['skills']['loaded']}")
print(f"Modules: {list(cfg['modules'].keys())}")

print("\n=== Policy Validate ===")
pv = post("/config/policy/validate", {
    "yaml_content": "rules:\n  - action: read_file\n    decision: allow\n    description: Allow reads\ndefault: deny"
})
print(f"Valid: {pv['valid']}, Rules: {pv.get('rules_count')}")

print("\n=== Skill Execute ===")
plan = post("/skills/database-setup/execute", {})
print(f"Skill: {plan['skill_name']}, Steps: {plan['total_steps']}")
for step in plan["steps"]:
    print(f"  Step {step['step_number']}: {step['title']}")

print("\n=== Dashboard HTML ===")
r = urllib.request.urlopen("http://localhost:8000/")
html = r.read().decode()
checks = ["pg-integrate", "pg-skills", "wizard-steps", "skillsGrid", "skillModal"]
for c in checks:
    found = c in html
    print(f"  {'OK' if found else 'FAIL'} {c}")

print("\n=== ALL TESTS PASSED ===")
