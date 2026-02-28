"""Quick dashboard verification script."""
import urllib.request
import json

BASE = "http://localhost:8000"

# Test 1: Dashboard HTML loads
print("=" * 50)
print("SHIELD DASHBOARD VERIFICATION")
print("=" * 50)

try:
    r = urllib.request.urlopen(BASE)
    html = r.read().decode("utf-8")
    print(f"[OK] Dashboard loaded - Status: {r.status}, Size: {len(html)} bytes")
    
    # Check key elements
    checks = {
        "Title": "Shield" in html,
        "CIBIL Page": "pg-cibil" in html,
        "Surveillance Page": "pg-surveillance" in html,
        "Shadow Page": "pg-shadow" in html,
        "Registry Page": "pg-registry" in html,
        "Overview Page": "pg-overview" in html,
        "Traces Page": "pg-traces" in html,
        "Skills Page": "pg-skills" in html,
        "Security Page": "pg-security" in html,
        "CSS Link": "styles.css" in html,
        "JS Link": "app.js" in html,
        "Chart.js": "chart.js" in html,
        "Particle Canvas": "particleCanvas" in html,
        "Brand v0.3.0": "v0.3.0" in html,
    }
    
    for name, ok in checks.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
    
    page_count = html.count('class="page"') + html.count("class='page'")
    nav_count = html.count('rail-link')
    print(f"\n  Pages found: {page_count}")
    print(f"  Nav links found: {nav_count}")
    
except Exception as e:
    print(f"[FAIL] Dashboard load failed: {e}")

# Test 2: Static CSS
print()
try:
    r = urllib.request.urlopen(BASE + "/static/styles.css")
    css = r.read().decode("utf-8")
    print(f"[OK] CSS loaded - {len(css)} bytes")
    css_checks = {
        "Glassmorphism": "glass-bg" in css,
        "Animations": "@keyframes" in css,
        "CIBIL Styles": "cibil-card" in css,
        "Connector Styles": "connector-card" in css,
        "Impact Bar": "impact-bar" in css,
        "Dark Theme": "#0a0a0f" in css,
        "Responsive": "@media" in css,
    }
    for name, ok in css_checks.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
except Exception as e:
    print(f"[FAIL] CSS load failed: {e}")

# Test 3: Static JS
print()
try:
    r = urllib.request.urlopen(BASE + "/static/app.js")
    js = r.read().decode("utf-8")
    print(f"[OK] JS loaded - {len(js)} bytes")
    js_checks = {
        "CIBIL Functions": "loadCibilScores" in js,
        "Surveillance Functions": "loadSurveillance" in js,
        "Shadow Functions": "loadShadowData" in js,
        "Registry Functions": "loadRegistryData" in js,
        "WebSocket": "connectWS" in js,
        "Particle System": "initParticles" in js,
        "Chart Init": "initCharts" in js,
        "Toast System": "toast(" in js,
    }
    for name, ok in js_checks.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
except Exception as e:
    print(f"[FAIL] JS load failed: {e}")

# Test 4: API endpoints
print()
print("API Endpoints:")
endpoints = [
    "/api/health",
    "/api/audit/logs",
    "/api/policies",
    "/api/cibil/scores",
    "/api/surveillance/status",
    "/api/shadow/results",
    "/api/registry/catalog",
    "/api/registry/connections",
]

for ep in endpoints:
    try:
        r = urllib.request.urlopen(BASE + ep)
        data = json.loads(r.read().decode("utf-8"))
        print(f"  [OK] {ep} -> {r.status}")
    except urllib.error.HTTPError as e:
        print(f"  [WARN] {ep} -> {e.code} (endpoint exists but returned error)")
    except Exception as e:
        print(f"  [FAIL] {ep} -> {e}")

print()
print("=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)
