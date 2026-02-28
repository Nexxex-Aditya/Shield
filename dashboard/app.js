/* ═══════════════════════════════════════════════════════════════
   SHIELD COMMAND CENTER — JavaScript Engine
   ═══════════════════════════════════════════════════════════════ */
(() => {
    'use strict';

    const API = '/api/v1';
    let ws = null;
    let traces = [];
    let pieChart, lineChart, cibilRadar, survLatencyChart;

    // ── Navigation ─────────────────────────────────────────────
    const titleMap = {
        overview: 'Command Center', traces: 'Traces', policies: 'Policies',
        agents: 'Agents', integrate: 'Integrate', skills: 'Skills',
        security: 'Security', escalations: 'Escalations',
        cibil: 'CIBIL Score', surveillance: 'Surveillance',
        shadow: 'Shadow Engine', registry: 'Registry',
        pipelines: 'Pipeline Workshop', models: 'AI Models'
    };

    document.querySelectorAll('.rail-link').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const page = link.dataset.page;
            document.querySelectorAll('.rail-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            const pg = document.getElementById('pg-' + page);
            if (pg) pg.classList.add('active');
            document.getElementById('topTitle').textContent = titleMap[page] || page;
            if (page === 'cibil') { loadCibilScores(); loadPipelineHealth(); }
            if (page === 'surveillance') loadSurveillance();
            if (page === 'shadow') loadShadowData();
            if (page === 'registry') loadRegistryData();
            if (page === 'pipelines') loadPipelines();
            if (page === 'models') loadModels();
            // close mobile rail
            document.getElementById('rail').classList.remove('open');
        });
    });

    document.getElementById('burger').addEventListener('click', () => {
        document.getElementById('rail').classList.toggle('open');
    });

    // ── Clock ──────────────────────────────────────────────────
    function updateClock() {
        const now = new Date();
        const el = document.getElementById('topClock');
        if (el) el.textContent = now.toLocaleTimeString('en-US', { hour12: false }) + ' UTC';
    }
    setInterval(updateClock, 1000);
    updateClock();

    // ── Toast ──────────────────────────────────────────────────
    function toast(msg, type = 'info') {
        const stack = document.getElementById('toasts');
        const el = document.createElement('div');
        el.className = 'toast ' + type;
        el.textContent = msg;
        stack.appendChild(el);
        setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 4000);
    }

    // ── API Helpers ────────────────────────────────────────────
    async function api(path, opts = {}) {
        try {
            const r = await fetch(API + path, { headers: { 'Content-Type': 'application/json' }, ...opts });
            if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
            return await r.json();
        } catch (e) {
            console.error('API error:', path, e);
            return null;
        }
    }

    // ── WebSocket ──────────────────────────────────────────────
    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/api/v1/ws`);
        ws.onopen = () => {
            document.getElementById('wsDot').className = 'ws-dot on';
            document.getElementById('wsLabel').textContent = 'Live';
        };
        ws.onclose = () => {
            document.getElementById('wsDot').className = 'ws-dot off';
            document.getElementById('wsLabel').textContent = 'Offline';
            setTimeout(connectWS, 3000);
        };
        ws.onmessage = e => {
            try {
                const data = JSON.parse(e.data);
                handleWSEvent(data);
            } catch (err) { console.warn('WS parse error', err); }
        };
    }

    function handleWSEvent(data) {
        if (data.type === 'trace') {
            traces.unshift(data.payload);
            if (traces.length > 500) traces.length = 500;
            renderTraces();
            updateOverview();
            addFeedItem(data.payload);
        } else if (data.type === 'escalation') {
            addEscalation(data.payload);
        } else if (data.type === 'anomaly') {
            addSurvAnomalyRow(data.payload);
        } else if (data.type === 'shadow_result') {
            addShadowRow(data.payload);
        } else if (data.type === 'pipeline_step_complete') {
            handlePipelineStepUpdate(data.payload || data.data);
        } else if (data.type === 'pipeline_completed' || data.type === 'pipeline_complete') {
            handlePipelineComplete(data.payload || data.data);
        } else if (data.type === 'pipeline_started' || data.type === 'pipeline_run_started') {
            if (typeof addPipeChatMsg === 'function') addPipeChatMsg('ai', `\u25b6 Pipeline run started: ${(data.data || data.payload || {}).name || (data.data || data.payload || {}).id || '?'}`);
        }
    }

    // ── Feed ───────────────────────────────────────────────────
    function addFeedItem(t) {
        const feed = document.getElementById('liveFeed');
        if (feed.querySelector('.muted')) feed.innerHTML = '';
        const item = document.createElement('div');
        item.className = 'feed-item';
        const ts = new Date(t.timestamp || Date.now()).toLocaleTimeString('en-US', { hour12: false });
        const dec = (t.decision || 'ALLOW').toUpperCase();
        item.innerHTML = `<span class="feed-time">${ts}</span>
            <span class="feed-text"><strong>${t.agent_id || 'agent'}</strong> &rarr; ${t.action || 'unknown'} &rarr;
            <span class="badge badge-${dec.toLowerCase()}">${dec}</span></span>`;
        feed.prepend(item);
        while (feed.children.length > 40) feed.lastChild.remove();
    }

    // ── Overview ───────────────────────────────────────────────
    async function loadOverview() {
        const data = await api('/audit');
        if (data && (data.logs || data.events)) {
            traces = data.logs || data.events || [];
            updateOverview();
            renderTraces();
        }
        initCharts();
        loadCibilMiniWidget();
        loadRegistryMiniWidget();
    }

    function updateOverview() {
        const total = traces.length;
        const allow = traces.filter(t => (t.decision || '').toUpperCase() === 'ALLOW').length;
        const deny = traces.filter(t => (t.decision || '').toUpperCase() === 'DENY').length;
        const esc = traces.filter(t => (t.decision || '').toUpperCase() === 'ESCALATE').length;
        setText('#mTotal .metric-num', total);
        setText('#mAllow .metric-num', allow);
        setText('#mDeny .metric-num', deny);
        setText('#mEscalate .metric-num', esc);
    }

    function setText(sel, val) {
        const el = document.querySelector(sel);
        if (el) el.textContent = val;
    }

    // ── Charts ─────────────────────────────────────────────────
    function initCharts() {
        const colors = { allow: '#22c55e', deny: '#dc2626', escalate: '#f59e0b' };
        const pieCtx = document.getElementById('chartPie');
        if (pieCtx && !pieChart) {
            pieChart = new Chart(pieCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Allowed', 'Denied', 'Escalated'],
                    datasets: [{ data: [0, 0, 0], backgroundColor: [colors.allow, colors.deny, colors.escalate], borderWidth: 0, spacing: 2 }]
                },
                options: {
                    cutout: '72%', responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'bottom', labels: { color: '#94a3b8', font: { size: 11 }, padding: 12, usePointStyle: true } }
                    }
                }
            });
        }
        updatePie();

        const lineCtx = document.getElementById('chartLine');
        if (lineCtx && !lineChart) {
            lineChart = new Chart(lineCtx, {
                type: 'line',
                data: {
                    labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
                    datasets: [{
                        label: 'Requests/hr',
                        data: Array(24).fill(0),
                        borderColor: '#dc2626',
                        backgroundColor: 'rgba(220,38,38,0.08)',
                        borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }
    }

    function updatePie() {
        if (!pieChart) return;
        const a = traces.filter(t => (t.decision || '').toUpperCase() === 'ALLOW').length;
        const d = traces.filter(t => (t.decision || '').toUpperCase() === 'DENY').length;
        const e = traces.filter(t => (t.decision || '').toUpperCase() === 'ESCALATE').length;
        pieChart.data.datasets[0].data = [a, d, e];
        pieChart.update();
    }

    // ── Traces ─────────────────────────────────────────────────
    function renderTraces() {
        const body = document.getElementById('tblBody');
        if (!body) return;
        const fDec = document.getElementById('fDecision')?.value || '';
        const fAg = document.getElementById('fAgent')?.value || '';
        const fAct = document.getElementById('fAction')?.value || '';
        let list = traces;
        if (fDec) list = list.filter(t => (t.decision || '').toUpperCase() === fDec);
        if (fAg) list = list.filter(t => t.agent_id === fAg);
        if (fAct) list = list.filter(t => (t.action || '').includes(fAct));

        body.innerHTML = list.slice(0, 100).map(t => {
            const dec = (t.decision || 'ALLOW').toUpperCase();
            const ts = new Date(t.timestamp || Date.now()).toLocaleString();
            const flags = (t.flags || []).map(f => `<span class="badge badge-deny">${f}</span>`).join(' ');
            return `<tr>
                <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-3)">${ts}</td>
                <td>${t.agent_id || '-'}</td>
                <td style="font-family:var(--font-mono)">${t.action || '-'}</td>
                <td><span class="badge badge-${dec.toLowerCase()}">${dec}</span></td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${t.reason || '-'}</td>
                <td>${flags || '-'}</td>
            </tr>`;
        }).join('');
    }

    ['fDecision', 'fAgent', 'fAction'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', renderTraces);
    });

    document.getElementById('exportBtn')?.addEventListener('click', () => {
        const rows = [['Timestamp', 'Agent', 'Action', 'Decision', 'Reason']];
        traces.forEach(t => rows.push([t.timestamp, t.agent_id, t.action, t.decision, t.reason]));
        const csv = rows.map(r => r.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'shield-traces.csv';
        a.click();
        toast('CSV exported', 'success');
    });

    // ── Policies ───────────────────────────────────────────────
    async function loadPolicies() {
        const data = await api('/policies');
        if (!data) return;
        const body = document.getElementById('polBody');
        if (!body) return;
        const rules = data.rules || [];
        body.innerHTML = rules.map(r => `<tr>
            <td style="font-family:var(--font-mono)">${r.action || '*'}</td>
            <td><span class="badge badge-${(r.decision || 'allow').toLowerCase()}">${r.decision || 'allow'}</span></td>
            <td>${r.description || '-'}</td>
            <td>${r.rate_limit || '-'}</td>
            <td>${r.sandbox ? 'Yes' : 'No'}</td>
            <td>${r.hits || 0}</td>
        </tr>`).join('');
    }

    document.getElementById('reloadBtn')?.addEventListener('click', async () => {
        await api('/policies/reload', { method: 'POST' });
        await loadPolicies();
        toast('Policies reloaded', 'success');
    });

    // ── Agents ─────────────────────────────────────────────────
    async function loadAgents() {
        const data = await api('/agents');
        const grid = document.getElementById('agentGrid');
        if (!data || !grid) return;
        const agents = data.agents || [];
        if (!agents.length) { grid.innerHTML = '<p class="muted">No agents registered yet.</p>'; return; }
        grid.innerHTML = agents.map(a => `<div class="agent-card">
            <h3 style="font-size:14px;font-weight:700;color:var(--text-0);margin-bottom:4px">${a.id || a.agent_id}</h3>
            <p style="font-size:12px;color:var(--text-2)">${a.description || 'No description'}</p>
            <p style="font-size:11px;color:var(--text-3);margin-top:8px">Runs: ${a.total_runs || 0}</p>
        </div>`).join('');
    }

    // ── Escalations ────────────────────────────────────────────
    function addEscalation(payload) {
        const queue = document.getElementById('escQueue');
        if (!queue) return;
        if (queue.querySelector('.muted')) queue.innerHTML = '';
        const el = document.createElement('div');
        el.className = 'card';
        el.style.marginBottom = '12px';
        el.innerHTML = `<div class="card-top"><h2 class="card-title">Escalation: ${payload.action || 'unknown'}</h2>
            <span class="badge badge-escalate">PENDING</span></div>
            <div class="card-body"><p>Agent: <strong>${payload.agent_id || '-'}</strong></p>
            <p>Reason: ${payload.reason || '-'}</p>
            <div class="form-actions"><button class="btn-primary btn-sm" onclick="this.closest('.card').remove()">Approve</button>
            <button class="btn-secondary btn-sm" onclick="this.closest('.card').remove()">Deny</button></div></div>`;
        queue.prepend(el);
        const badge = document.getElementById('escBadge');
        if (badge) { badge.style.display = ''; badge.textContent = queue.children.length; }
    }

    // ── Security Tabs ──────────────────────────────────────────
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const pane = tab.dataset.tab;
            tab.closest('.tab-row').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            tab.closest('section').querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(pane)?.classList.add('active');
        });
    });

    document.getElementById('narrBtn')?.addEventListener('click', async () => {
        const body = document.getElementById('narrBody');
        if (!body) return;
        body.innerHTML = '<p class="muted">Generating narrative...</p>';
        const data = await api('/narrative/overview');
        body.innerHTML = data ? `<p>${data.narrative || data.summary || 'No narrative generated.'}</p>` : '<p class="muted">Generation failed.</p>';
    });

    // ── Integrate Wizard ───────────────────────────────────────
    let wizStep = 1;
    const wizMax = 5;
    function updateWizard() {
        document.querySelectorAll('.wiz-step').forEach(s => {
            const n = parseInt(s.dataset.step);
            s.classList.toggle('active', n === wizStep);
            s.classList.toggle('done', n < wizStep);
        });
        document.querySelectorAll('.wiz-panel').forEach((p, i) => p.classList.toggle('active', i + 1 === wizStep));
        document.getElementById('wizPrev').disabled = wizStep === 1;
        document.getElementById('wizNext').textContent = wizStep === wizMax ? 'Finish' : 'Next \u2192';
    }
    document.getElementById('wizNext')?.addEventListener('click', () => { if (wizStep < wizMax) { wizStep++; updateWizard(); } });
    document.getElementById('wizPrev')?.addEventListener('click', () => { if (wizStep > 1) { wizStep--; updateWizard(); } });

    document.getElementById('dbTestBtn')?.addEventListener('click', async () => {
        const result = document.getElementById('dbTestResult');
        result.textContent = 'Testing...'; result.className = 'test-result';
        const data = await api('/config/database/test', { method: 'POST', body: JSON.stringify({ uri: document.getElementById('dbUri')?.value || '' }) });
        if (data?.ok) { result.textContent = 'Connected!'; result.className = 'test-result ok'; }
        else { result.textContent = 'Failed'; result.className = 'test-result err'; }
    });

    document.getElementById('healthCheckBtn')?.addEventListener('click', async () => {
        const data = await api('/config/system');
        if (!data) return;
        const checks = { hDb: data.database, hGw: data.gateway, hSkills: data.skills, hPolicy: data.policy, hSec: data.security, hShadow: data.shadow, hSurv: data.surveillance, hCibil: data.cibil, hReg: data.registry };
        Object.entries(checks).forEach(([id, val]) => {
            const el = document.getElementById(id);
            if (!el) return;
            const ok = val === true || val === 'ok' || val === 'healthy';
            el.textContent = ok ? 'OK' : (val || 'N/A');
            const dot = el.closest('.health-row')?.querySelector('.health-dot');
            if (dot) dot.className = 'health-dot ' + (ok ? 'ok' : 'fail');
        });
    });

    // ── Skills ─────────────────────────────────────────────────
    async function loadSkills() {
        const data = await api('/skills');
        const grid = document.getElementById('skillsGrid');
        if (!data || !grid) return;
        const skills = data.skills || [];
        const badge = document.getElementById('skillsBadge');
        if (badge && skills.length) { badge.style.display = ''; badge.textContent = skills.length; }
        if (!skills.length) { grid.innerHTML = '<p class="muted">No skills yet. Create one!</p>'; return; }
        grid.innerHTML = skills.map(s => `<div class="skill-card">
            <h3>${s.name}</h3>
            <p>${s.description || 'No description'}</p>
            <div class="skill-tags">${(s.tags || []).map(t => `<span class="skill-tag">${t}</span>`).join('')}</div>
        </div>`).join('');
    }

    // Skill modals
    document.getElementById('skillCreateBtn')?.addEventListener('click', () => {
        document.getElementById('skillModal').style.display = '';
        document.getElementById('skillModalTitle').textContent = 'Create Skill';
    });
    document.getElementById('skillModalClose')?.addEventListener('click', () => document.getElementById('skillModal').style.display = 'none');
    document.getElementById('skillImportBtn')?.addEventListener('click', () => document.getElementById('importModal').style.display = '');
    document.getElementById('importModalClose')?.addEventListener('click', () => document.getElementById('importModal').style.display = 'none');
    document.getElementById('yamlModalClose')?.addEventListener('click', () => document.getElementById('yamlModal').style.display = 'none');

    document.getElementById('addStepBtn')?.addEventListener('click', () => {
        const editor = document.getElementById('stepsEditor');
        const idx = editor.children.length + 1;
        const step = document.createElement('div');
        step.className = 'form-row';
        step.style.background = 'var(--bg-3)'; step.style.padding = '12px'; step.style.borderRadius = '8px';
        step.innerHTML = `<label class="form-sublabel">Step ${idx}</label>
            <input class="form-input step-title" placeholder="Step title" style="margin-bottom:6px">
            <textarea class="form-input step-instr" rows="2" placeholder="Instruction"></textarea>`;
        editor.appendChild(step);
    });

    document.getElementById('skillSaveBtn')?.addEventListener('click', async () => {
        const steps = Array.from(document.querySelectorAll('#stepsEditor > div')).map(s => ({
            title: s.querySelector('.step-title')?.value || '',
            instruction: s.querySelector('.step-instr')?.value || ''
        }));
        const skill = {
            name: document.getElementById('skName')?.value || '',
            description: document.getElementById('skDesc')?.value || '',
            version: document.getElementById('skVersion')?.value || '1.0',
            tags: (document.getElementById('skTags')?.value || '').split(',').map(t => t.trim()).filter(Boolean),
            steps: steps,
            permissions: {
                allowed_tools: (document.getElementById('skAllowed')?.value || '').split(',').map(t => t.trim()).filter(Boolean),
                blocked_tools: (document.getElementById('skBlocked')?.value || '').split(',').map(t => t.trim()).filter(Boolean),
                require_approval: document.getElementById('skApproval')?.checked || false,
                sandbox: document.getElementById('skSandbox')?.checked || false
            }
        };
        await api('/skills', { method: 'POST', body: JSON.stringify(skill) });
        document.getElementById('skillModal').style.display = 'none';
        await loadSkills();
        toast('Skill created', 'success');
    });

    // ── CIBIL Score ────────────────────────────────────────────
    async function loadCibilScores() {
        const data = await api('/api/cibil/scores');
        const grid = document.getElementById('cibilModelsGrid');
        if (!grid) return;
        if (!data || !data.models || !data.models.length) {
            grid.innerHTML = '<p class="muted">No models tracked yet. Models will appear after tool calls flow through the pipeline.</p>';
            return;
        }
        grid.innerHTML = data.models.map(m => {
            const score = Math.round((m.overall_score || 0) * 100);
            const grade = score >= 80 ? 'a' : score >= 60 ? 'b' : score >= 40 ? 'c' : score >= 20 ? 'd' : 'f';
            const gradeChar = grade.toUpperCase();
            const barColor = grade === 'a' ? '#22c55e' : grade === 'b' ? '#3b82f6' : grade === 'c' ? '#f59e0b' : grade === 'd' ? '#f97316' : '#dc2626';
            return `<div class="cibil-card grade-${grade}" onclick="window._showCibilDetail('${m.model_id}')">
                <div class="cibil-header">
                    <span class="cibil-model-name">${m.model_id}</span>
                    <span class="cibil-grade grade-${grade}">${gradeChar}</span>
                </div>
                <div class="cibil-score-bar"><div class="cibil-score-fill" style="width:${score}%;background:${barColor}"></div></div>
                <div class="cibil-meta">
                    <span class="cibil-score-num">${score}/100</span>
                    <span>${m.total_interactions || 0} interactions</span>
                </div>
            </div>`;
        }).join('');
    }

    async function loadCibilMiniWidget() {
        const data = await api('/api/cibil/scores');
        const list = document.getElementById('cibilMiniList');
        if (!list) return;
        if (!data || !data.models || !data.models.length) {
            list.innerHTML = '<p class="muted">No models tracked yet.</p>';
            return;
        }
        list.innerHTML = data.models.slice(0, 5).map(m => {
            const score = Math.round((m.overall_score || 0) * 100);
            const grade = score >= 80 ? 'a' : score >= 60 ? 'b' : score >= 40 ? 'c' : score >= 20 ? 'd' : 'f';
            const bg = grade === 'a' ? 'rgba(34,197,94,.15)' : grade === 'b' ? 'rgba(59,130,246,.15)' : 'rgba(245,158,11,.15)';
            const fg = grade === 'a' ? '#22c55e' : grade === 'b' ? '#3b82f6' : '#f59e0b';
            return `<div class="cibil-mini-item">
                <span class="cibil-mini-name">${m.model_id}</span>
                <span class="cibil-mini-score" style="background:${bg};color:${fg}">${score}</span>
            </div>`;
        }).join('');
    }

    window._showCibilDetail = async function (modelId) {
        document.getElementById('cibilDetail').style.display = '';
        document.getElementById('cibilDetailTitle').textContent = 'Model: ' + modelId;
        // Load report
        const report = await api('/cibil/models/' + encodeURIComponent(modelId) + '/report');
        const body = document.getElementById('cibilReportBody');
        if (report && body) {
            body.innerHTML = `<div style="display:flex;flex-direction:column;gap:12px">
                <div><strong>Overall Score:</strong> <span style="font-family:var(--font-mono);font-size:18px;font-weight:800">${Math.round((report.overall_score || 0) * 100)}/100</span></div>
                <div><strong>Strengths:</strong> ${(report.strengths || []).join(', ') || 'None identified'}</div>
                <div><strong>Weaknesses:</strong> ${(report.weaknesses || []).join(', ') || 'None identified'}</div>
                <div><strong>Insights:</strong></div>
                ${(report.insights || []).map(i => `<p style="color:var(--text-2);font-size:12px;padding-left:12px;border-left:2px solid var(--purple)">- ${i}</p>`).join('')}
            </div>`;
        }
        // Radar chart
        const cats = report?.categories || {};
        const labels = Object.keys(cats);
        const values = Object.values(cats).map(v => typeof v === 'number' ? v * 100 : (v?.score || 0) * 100);
        const rCtx = document.getElementById('cibilRadar');
        if (rCtx && labels.length) {
            if (cibilRadar) cibilRadar.destroy();
            cibilRadar = new Chart(rCtx, {
                type: 'radar',
                data: {
                    labels: labels,
                    datasets: [{ label: modelId, data: values, borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.15)', borderWidth: 2, pointRadius: 3 }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: { r: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.06)' }, angleLines: { color: 'rgba(255,255,255,0.06)' }, pointLabels: { color: '#94a3b8', font: { size: 10 } }, ticks: { display: false } } },
                    plugins: { legend: { display: false } }
                }
            });
        }
        // Recommendations
        const recs = await api('/recommendations/report/' + encodeURIComponent(modelId));
        const recsEl = document.getElementById('cibilRecommendations');
        if (recs && recsEl) {
            const items = recs.recommendations || [];
            recsEl.innerHTML = items.length ? items.map(r => `<div style="padding:10px 0;border-bottom:1px solid var(--glass-border)">
                <span class="badge badge-${r.type === 'warning' ? 'warn' : 'proceed'}">${r.type || 'tip'}</span>
                <span style="margin-left:8px;color:var(--text-1)">${r.message || r.text || ''}</span>
            </div>`).join('') : '<p class="muted">No recommendations.</p>';
        }
    };

    document.getElementById('cibilCloseDetail')?.addEventListener('click', () => {
        document.getElementById('cibilDetail').style.display = 'none';
    });
    document.getElementById('cibilRefresh')?.addEventListener('click', loadCibilScores);

    // ── Surveillance ───────────────────────────────────────────
    async function loadSurveillance() {
        const data = await api('/surveillance/tools');
        const grid = document.getElementById('survToolsGrid');
        if (!grid) return;
        if (!data || !data.tools || !data.tools.length) {
            grid.innerHTML = '<p class="muted">No tools being monitored. Tools are tracked as requests flow through Shield.</p>';
        } else {
            grid.innerHTML = '<div class="surv-tool-grid">' + data.tools.map(t => `<div class="surv-tool-card">
                <div class="surv-tool-name">${t.tool_name || t.tool_id}</div>
                <div class="surv-tool-stat">Avg latency: <strong>${t.avg_latency ? t.avg_latency.toFixed(0) + 'ms' : 'N/A'}</strong></div>
                <div class="surv-tool-stat">Calls: <strong>${t.call_count || 0}</strong></div>
                <div class="surv-tool-stat">Errors: <strong>${t.error_count || 0}</strong></div>
            </div>`).join('') + '</div>';
        }
        loadSurvAnomalies();
    }

    async function loadSurvAnomalies() {
        const data = await api('/surveillance/anomalies');
        const body = document.getElementById('survAnomalyBody');
        if (!body) return;
        if (!data || !data.anomalies || !data.anomalies.length) {
            body.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-3);font-style:italic">No anomalies detected</td></tr>';
            return;
        }
        body.innerHTML = data.anomalies.map(a => {
            const sev = (a.severity || 'low').toLowerCase();
            return `<tr>
                <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-3)">${new Date(a.timestamp || Date.now()).toLocaleString()}</td>
                <td>${a.tool_name || '-'}</td>
                <td>${a.anomaly_type || '-'}</td>
                <td><span class="badge badge-${sev}">${sev.toUpperCase()}</span></td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${a.detail || '-'}</td>
                <td style="font-family:var(--font-mono)">${a.deviation ? a.deviation.toFixed(2) + 'x' : '-'}</td>
            </tr>`;
        }).join('');
    }

    function addSurvAnomalyRow(a) {
        const body = document.getElementById('survAnomalyBody');
        if (!body) return;
        if (body.querySelector('td[colspan]')) body.innerHTML = '';
        const sev = (a.severity || 'low').toLowerCase();
        const row = document.createElement('tr');
        row.innerHTML = `<td style="font-family:var(--font-mono);font-size:11px;color:var(--text-3)">${new Date().toLocaleString()}</td>
            <td>${a.tool_name || '-'}</td><td>${a.anomaly_type || '-'}</td>
            <td><span class="badge badge-${sev}">${sev.toUpperCase()}</span></td>
            <td>${a.detail || '-'}</td><td>${a.deviation ? a.deviation.toFixed(2) + 'x' : '-'}</td>`;
        body.prepend(row);
    }

    document.getElementById('survRefresh')?.addEventListener('click', loadSurveillance);

    // ── Shadow Engine ──────────────────────────────────────────
    async function loadShadowData() {
        const data = await api('/shadow/results');
        const body = document.getElementById('shadowBody');
        if (!body) return;
        if (!data || !data.results || !data.results.length) {
            body.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-3);font-style:italic">No shadow checks yet</td></tr>';
            return;
        }
        const results = data.results;
        setText('#shTotal .metric-num', results.length);
        setText('#shProceed .metric-num', results.filter(r => r.verdict === 'proceed').length);
        setText('#shWarn .metric-num', results.filter(r => r.verdict === 'warn').length);
        setText('#shEscalate .metric-num', results.filter(r => r.verdict === 'escalate').length);
        setText('#shBlock .metric-num', results.filter(r => r.verdict === 'block').length);

        body.innerHTML = results.slice(0, 50).map(r => {
            const v = (r.verdict || 'proceed').toLowerCase();
            const impact = r.impact_score || 0;
            const impClass = impact >= 0.75 ? 'crit' : impact >= 0.5 ? 'high' : impact >= 0.25 ? 'med' : 'low';
            return `<tr>
                <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-3)">${new Date(r.timestamp || Date.now()).toLocaleString()}</td>
                <td>${r.tool_name || '-'}</td>
                <td><div class="impact-bar"><div class="impact-fill impact-${impClass}" style="width:${impact * 100}%"></div></div>${(impact * 100).toFixed(0)}%</td>
                <td><span class="badge badge-${v}">${v.toUpperCase()}</span></td>
                <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis">${(r.side_effects || []).join(', ') || 'None'}</td>
                <td style="font-family:var(--font-mono)">${r.duration_ms ? r.duration_ms + 'ms' : '-'}</td>
            </tr>`;
        }).join('');
    }

    function addShadowRow(r) {
        const body = document.getElementById('shadowBody');
        if (!body) return;
        if (body.querySelector('td[colspan]')) body.innerHTML = '';
        const v = (r.verdict || 'proceed').toLowerCase();
        const impact = r.impact_score || 0;
        const impClass = impact >= 0.75 ? 'crit' : impact >= 0.5 ? 'high' : impact >= 0.25 ? 'med' : 'low';
        const row = document.createElement('tr');
        row.innerHTML = `<td style="font-family:var(--font-mono);font-size:11px;color:var(--text-3)">${new Date().toLocaleString()}</td>
            <td>${r.tool_name || '-'}</td>
            <td><div class="impact-bar"><div class="impact-fill impact-${impClass}" style="width:${impact * 100}%"></div></div>${(impact * 100).toFixed(0)}%</td>
            <td><span class="badge badge-${v}">${v.toUpperCase()}</span></td>
            <td>${(r.side_effects || []).join(', ') || 'None'}</td>
            <td>${r.duration_ms ? r.duration_ms + 'ms' : '-'}</td>`;
        body.prepend(row);
    }

    document.getElementById('shadowRefresh')?.addEventListener('click', loadShadowData);

    // ── Registry ───────────────────────────────────────────────
    async function loadRegistryData() {
        const catalog = await api('/registry/connectors');
        const grid = document.getElementById('connectorGrid');
        if (grid && catalog && catalog.connectors) {
            grid.innerHTML = catalog.connectors.map(c => `<div class="connector-card" onclick="window._connectService('${c.id}', '${c.name}')">
                <div class="connector-icon">${c.icon || '🔗'}</div>
                <div class="connector-name">${c.name}</div>
                <div class="connector-cat">${c.category || 'general'}</div>
            </div>`).join('');
        }
        const conns = await api('/registry/connections');
        const connList = document.getElementById('regConnectionsList');
        const connCount = document.getElementById('regConnCount');
        if (connList && conns && conns.connections) {
            if (connCount) connCount.textContent = conns.connections.length;
            if (!conns.connections.length) {
                connList.innerHTML = '<p class="muted">No active connections. Click a connector below to connect.</p>';
            } else {
                connList.innerHTML = conns.connections.map(c => `<div class="connection-item">
                    <span class="conn-status-dot ${c.healthy !== false ? 'healthy' : 'unhealthy'}"></span>
                    <div class="conn-info">
                        <div class="conn-name">${c.name || c.connector_id}</div>
                        <div class="conn-meta">${c.connector_id} &middot; ${c.category || '-'}</div>
                    </div>
                    <div class="conn-actions">
                        <button class="text-btn" onclick="window._disconnectService('${c.connection_id}')">Disconnect</button>
                    </div>
                </div>`).join('');
            }
        }
        const suggestions = await api('/registry/suggestions');
        const sugCard = document.getElementById('regSuggestionsCard');
        const sugBody = document.getElementById('regSuggestions');
        if (sugCard && sugBody && suggestions && suggestions.suggestions && suggestions.suggestions.length) {
            sugCard.style.display = '';
            sugBody.innerHTML = suggestions.suggestions.map(s => `<div style="padding:8px 0;border-bottom:1px solid var(--glass-border)">
                <strong style="color:var(--text-0)">${s.name || s.connector_id}</strong>
                <span style="color:var(--text-2);margin-left:8px">${s.reason || ''}</span>
            </div>`).join('');
        }
    }

    async function loadRegistryMiniWidget() {
        const data = await api('/registry/connections');
        const list = document.getElementById('registryMiniList');
        if (!list) return;
        if (!data || !data.connections || !data.connections.length) {
            list.innerHTML = '<p class="muted">No services connected.</p>';
            return;
        }
        list.innerHTML = data.connections.slice(0, 5).map(c => `<div class="connection-item" style="margin-bottom:4px">
            <span class="conn-status-dot ${c.healthy !== false ? 'healthy' : 'unhealthy'}"></span>
            <div class="conn-info"><div class="conn-name">${c.name || c.connector_id}</div></div>
        </div>`).join('');
    }

    window._connectService = function (connectorId, name) {
        const modal = document.getElementById('connectModal');
        if (!modal) return;
        modal.style.display = '';
        document.getElementById('connectModalTitle').textContent = 'Connect: ' + name;
        document.getElementById('connectModalBody').innerHTML = `<div class="form-row">
            <label class="form-label">Connection Name</label>
            <input id="connName" class="form-input" type="text" placeholder="my-${connectorId}" spellcheck="false">
        </div><div class="form-row">
            <label class="form-label">Configuration (JSON)</label>
            <textarea id="connConfig" class="form-textarea" rows="6" spellcheck="false" placeholder='{"host": "localhost", "port": 5432}'></textarea>
        </div>
        <input type="hidden" id="connectorId" value="${connectorId}">`;
    };
    document.getElementById('connectModalClose')?.addEventListener('click', () => document.getElementById('connectModal').style.display = 'none');
    document.getElementById('connectCancelBtn')?.addEventListener('click', () => document.getElementById('connectModal').style.display = 'none');
    document.getElementById('connectSaveBtn')?.addEventListener('click', async () => {
        const connectorId = document.getElementById('connectorId')?.value;
        const name = document.getElementById('connName')?.value;
        let config = {};
        try { config = JSON.parse(document.getElementById('connConfig')?.value || '{}'); } catch (e) { toast('Invalid JSON config', 'error'); return; }
        await api('/registry/connect', { method: 'POST', body: JSON.stringify({ connector_id: connectorId, instance_name: name, config: config }) });
        document.getElementById('connectModal').style.display = 'none';
        await loadRegistryData();
        toast('Service connected', 'success');
    });

    window._disconnectService = async function (connectionId) {
        await api('/registry/connections/' + encodeURIComponent(connectionId), { method: 'DELETE' });
        await loadRegistryData();
        toast('Service disconnected', 'info');
    };

    document.getElementById('regRefresh')?.addEventListener('click', loadRegistryData);

    // ── Particle Background ────────────────────────────────────
    function initParticles() {
        const canvas = document.getElementById('particleCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let w, h;
        const particles = [];
        const count = 60;

        function resize() {
            w = canvas.width = window.innerWidth;
            h = canvas.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);

        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * w, y: Math.random() * h,
                vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
                r: Math.random() * 1.5 + 0.5,
                a: Math.random() * 0.3 + 0.1
            });
        }

        function draw() {
            ctx.clearRect(0, 0, w, h);
            particles.forEach(p => {
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0) p.x = w;
                if (p.x > w) p.x = 0;
                if (p.y < 0) p.y = h;
                if (p.y > h) p.y = 0;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(220, 38, 38, ${p.a})`;
                ctx.fill();
            });
            // Connect nearby particles
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(220, 38, 38, ${0.06 * (1 - dist / 120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
    }

    // ── Refresh Button ─────────────────────────────────────────
    document.getElementById('refreshBtn')?.addEventListener('click', () => {
        loadOverview();
        loadPolicies();
        loadAgents();
        loadSkills();
        toast('Data refreshed', 'info');
    });

    // ── Init ───────────────────────────────────────────────────
    function init() {
        initParticles();
        connectWS();
        loadOverview();
        loadPolicies();
        loadAgents();
        loadSkills();
    }

    // ── Pipeline Editor ────────────────────────────────────────
    let currentPipeline = null;
    let pipeZoom = 1;
    let pipeOffsetX = 0, pipeOffsetY = 0;
    let pipeDragging = false, pipeLastX = 0, pipeLastY = 0;

    async function loadPipelines() {
        // Load templates
        const tmplRow = document.getElementById('pipeTemplatesRow');
        if (tmplRow && !tmplRow.dataset.loaded) {
            const tmplData = await api('/pipelines/templates');
            if (tmplData && tmplData.templates) {
                tmplRow.innerHTML = '<span style="color:var(--text-3);font-size:12px;margin-right:4px">Templates:</span>' +
                    tmplData.templates.map(t => `<button class="btn-secondary btn-sm" style="font-size:11px;padding:4px 10px" onclick="window._pipeUseTemplate('${t.name}')">${t.name}</button>`).join('');
                tmplRow.dataset.loaded = '1';
            }
        }
        // Load saved pipelines
        const data = await api('/pipelines');
        const body = document.getElementById('pipeTableBody');
        const count = document.getElementById('pipeCount');
        if (!body) return;
        const pipelines = data?.pipelines || [];
        if (count) count.textContent = pipelines.length;
        if (!pipelines.length) {
            body.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-3)">No pipelines yet. Describe one above!</td></tr>';
            return;
        }
        body.innerHTML = pipelines.map(p => {
            const statusCls = p.status === 'completed' ? 'allow' : p.status === 'failed' ? 'deny' : p.status === 'running' ? 'escalate' : 'proceed';
            return `<tr>
                <td style="font-weight:600;color:var(--text-0)">${p.name}</td>
                <td>${p.steps?.length || p.step_count || 0}</td>
                <td><span class="badge badge-${statusCls}">${(p.status || 'draft').toUpperCase()}</span></td>
                <td>${p.run_count || 0}</td>
                <td style="font-family:var(--font-mono)">${p.last_duration_ms ? p.last_duration_ms.toFixed(0) + 'ms' : '—'}</td>
                <td>
                    <button class="text-btn" onclick="window._pipeView('${p.id}')">View</button>
                    <button class="text-btn" onclick="window._pipeRun('${p.id}')">Run</button>
                    <button class="text-btn red" onclick="window._pipeDelete('${p.id}')">Delete</button>
                </td>
            </tr>`;
        }).join('');
    }

    // Compile pipeline + post to chat
    async function compilePipelineFromText(desc, isChat = false) {
        if (!desc) { toast('Enter a pipeline description', 'error'); return; }
        // Add user message to chat
        addPipeChatMsg('user', desc);
        toast('Compiling pipeline...', 'info');
        const context = currentPipeline ? { refine_from: currentPipeline.id, previous_steps: currentPipeline.steps } : {};
        const data = await api('/pipelines/compile', { method: 'POST', body: JSON.stringify({ description: desc, context }) });
        if (data?.pipeline) {
            currentPipeline = data.pipeline;
            renderPipelineDAG(currentPipeline);
            showPipelineInfo(currentPipeline);
            await loadPipelines();
            const stepNames = (currentPipeline.steps || []).map(s => s.name || s.id).join(' → ');
            addPipeChatMsg('ai', `✅ Compiled **${currentPipeline.name}** with ${currentPipeline.steps?.length || 0} steps:\n${stepNames}`);
            toast('Pipeline compiled: ' + currentPipeline.name, 'success');
        } else {
            addPipeChatMsg('ai', '❌ Compilation failed. Try rephrasing or adding more detail.');
            toast('Compilation failed', 'error');
        }
    }

    function addPipeChatMsg(role, text) {
        const container = document.getElementById('pipeChatMessages');
        if (!container) return;
        const el = document.createElement('div');
        el.className = 'pipe-chat-msg ' + role;
        el.style.cssText = role === 'user'
            ? 'align-self:flex-end;background:rgba(220,38,38,0.15);border:1px solid rgba(220,38,38,0.3);border-radius:12px 12px 2px 12px;padding:8px 12px;max-width:80%;font-size:13px;color:var(--text-0)'
            : 'align-self:flex-start;background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2);border-radius:12px 12px 12px 2px;padding:8px 12px;max-width:80%;font-size:13px;color:var(--text-1)';
        // Simple markdown bold
        el.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
    }

    document.getElementById('pipeCompileBtn')?.addEventListener('click', () => {
        const desc = document.getElementById('pipeDescription')?.value;
        compilePipelineFromText(desc);
        document.getElementById('pipeDescription').value = '';
    });

    // Chat send
    document.getElementById('pipeChatSendBtn')?.addEventListener('click', () => {
        const input = document.getElementById('pipeChatInput');
        const text = input?.value?.trim();
        if (text) { compilePipelineFromText(text, true); input.value = ''; }
    });
    document.getElementById('pipeChatInput')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') document.getElementById('pipeChatSendBtn')?.click();
    });
    document.getElementById('pipeChatClearBtn')?.addEventListener('click', () => {
        const c = document.getElementById('pipeChatMessages');
        if (c) c.innerHTML = '<div class="pipe-chat-msg ai"><span style="opacity:.5">Describe a workflow and I\'ll compile it into a pipeline. Ask follow-up questions to refine it.</span></div>';
    });

    // Template buttons
    window._pipeUseTemplate = function (name) {
        document.getElementById('pipeDescription').value = name.replace(/([A-Z])/g, ' $1').trim();
        document.getElementById('pipeCompileBtn').click();
    };

    // View pipeline
    window._pipeView = async function (id) {
        const data = await api('/pipelines/' + id);
        if (data) {
            currentPipeline = data;
            renderPipelineDAG(data);
            showPipelineInfo(data);
        }
    };

    // Run pipeline
    window._pipeRun = async function (id) {
        toast('Running pipeline...', 'info');
        const data = await api('/pipelines/run', { method: 'POST', body: JSON.stringify({ pipeline_id: id, context: {} }) });
        if (data) {
            toast(`Pipeline ${data.status}: ${data.steps_completed}/${data.total_steps} in ${data.duration_ms?.toFixed(0) || '?'}ms`, data.status === 'completed' ? 'success' : 'error');
            await loadPipelines();
            if (currentPipeline?.id === id) {
                const fresh = await api('/pipelines/' + id);
                if (fresh) { currentPipeline = fresh; renderPipelineDAG(fresh); showPipelineInfo(fresh); }
            }
        }
    };

    // Delete pipeline
    window._pipeDelete = async function (id) {
        await api('/pipelines/' + id, { method: 'DELETE' });
        toast('Pipeline deleted', 'info');
        await loadPipelines();
    };

    function showPipelineInfo(p) {
        const info = document.getElementById('pipeInfo');
        if (!info) return;
        info.style.display = '';
        document.getElementById('pipeInfoId').textContent = p.id;
        document.getElementById('pipeInfoSteps').textContent = p.steps?.length || 0;
        document.getElementById('pipeInfoStatus').textContent = (p.status || 'draft').toUpperCase();
        document.getElementById('pipeInfoRuns').textContent = p.run_count || 0;
        document.getElementById('pipeInfoLast').textContent = p.last_run || '—';
        document.getElementById('pipeCanvasTitle').textContent = p.name || 'Pipeline DAG';
    }

    // ── Canvas DAG Renderer ─────────────────────────────────────
    const STEP_COLORS = {
        tool_call: '#dc2626', llm_call: '#8b5cf6', transform: '#3b82f6',
        condition: '#f59e0b', webhook: '#06b6d4', delay: '#64748b',
        human_review: '#22c55e', parallel: '#ec4899'
    };
    const NODE_W = 160, NODE_H = 52, NODE_R = 12;

    function renderPipelineDAG(pipeline) {
        const wrap = document.getElementById('pipeCanvasWrap');
        const canvas = document.getElementById('pipeCanvas');
        const placeholder = document.getElementById('pipeEmptyPlaceholder');
        if (!canvas || !wrap) return;

        const steps = pipeline.steps || [];
        if (!steps.length) { if (placeholder) placeholder.style.display = ''; return; }
        if (placeholder) placeholder.style.display = 'none';

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = wrap.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.scale(dpr, dpr);

        // Layout: assign positions via topological levels
        const idMap = {};
        steps.forEach((s, i) => { idMap[s.id] = i; });
        const levels = [];
        const assigned = new Set();
        // Find entry nodes
        let current = steps.filter(s => !s.depends_on || !s.depends_on.length);
        while (current.length > 0) {
            levels.push(current.map(s => s.id));
            current.forEach(s => assigned.add(s.id));
            const next = [];
            steps.forEach(s => {
                if (!assigned.has(s.id) && s.depends_on && s.depends_on.every(d => assigned.has(d))) {
                    next.push(s);
                }
            });
            current = next;
            if (current.length === 0) {
                // Catch any remaining unassigned
                const remaining = steps.filter(s => !assigned.has(s.id));
                if (remaining.length) { levels.push(remaining.map(s => s.id)); remaining.forEach(s => assigned.add(s.id)); }
            }
        }

        const xGap = 200, yGap = 80;
        const totalH = levels.length * (NODE_H + yGap);
        const nodePositions = {};
        levels.forEach((level, li) => {
            const totalW = level.length * NODE_W + (level.length - 1) * 40;
            const startX = (rect.width / pipeZoom - totalW) / 2;
            level.forEach((sid, si) => {
                nodePositions[sid] = {
                    x: startX + si * (NODE_W + 40) + pipeOffsetX,
                    y: 40 + li * (NODE_H + yGap) + pipeOffsetY
                };
            });
        });

        // Clear & draw
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.save();
        ctx.scale(pipeZoom, pipeZoom);

        // Draw edges
        steps.forEach(s => {
            const from = nodePositions[s.id];
            if (!from) return;
            const deps = [...(s.depends_on || []), ...(s.on_success || [])];
            deps.forEach(depId => {
                const to = nodePositions[depId];
                if (!to) return;
                const [fromPos, toPos] = s.depends_on?.includes(depId)
                    ? [to, from] : [from, to];
                ctx.beginPath();
                ctx.moveTo(fromPos.x + NODE_W / 2, fromPos.y + NODE_H);
                const midY = (fromPos.y + NODE_H + toPos.y) / 2;
                ctx.bezierCurveTo(
                    fromPos.x + NODE_W / 2, midY,
                    toPos.x + NODE_W / 2, midY,
                    toPos.x + NODE_W / 2, toPos.y
                );
                ctx.strokeStyle = 'rgba(255,255,255,0.15)';
                ctx.lineWidth = 2;
                ctx.stroke();
                // Arrow head
                const angle = Math.atan2(toPos.y - midY, toPos.x + NODE_W / 2 - (toPos.x + NODE_W / 2));
                ctx.beginPath();
                ctx.moveTo(toPos.x + NODE_W / 2, toPos.y);
                ctx.lineTo(toPos.x + NODE_W / 2 - 5, toPos.y - 8);
                ctx.lineTo(toPos.x + NODE_W / 2 + 5, toPos.y - 8);
                ctx.fillStyle = 'rgba(255,255,255,0.2)';
                ctx.fill();
            });
        });

        // Draw nodes
        steps.forEach(s => {
            const pos = nodePositions[s.id];
            if (!pos) return;
            const color = STEP_COLORS[s.type] || '#64748b';
            const statusColor = s.status === 'completed' ? '#22c55e' : s.status === 'failed' ? '#dc2626' : s.status === 'running' ? '#f59e0b' : null;

            // Node bg
            ctx.beginPath();
            ctx.roundRect(pos.x, pos.y, NODE_W, NODE_H, NODE_R);
            ctx.fillStyle = `${color}22`;
            ctx.fill();
            ctx.strokeStyle = statusColor || color;
            ctx.lineWidth = statusColor ? 2.5 : 1.5;
            ctx.stroke();

            // Type badge
            ctx.fillStyle = color;
            ctx.font = '600 9px Inter, sans-serif';
            const typeLabel = (s.type || 'tool_call').toUpperCase().replace('_', ' ');
            ctx.fillText(typeLabel, pos.x + 10, pos.y + 16);

            // Name
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '500 12px Inter, sans-serif';
            const name = (s.name || s.id).slice(0, 20);
            ctx.fillText(name, pos.x + 10, pos.y + 36);

            // Status dot
            if (statusColor) {
                ctx.beginPath();
                ctx.arc(pos.x + NODE_W - 14, pos.y + 14, 4, 0, Math.PI * 2);
                ctx.fillStyle = statusColor;
                ctx.fill();
            }
        });

        ctx.restore();

        // Store positions for click detection
        canvas._nodePositions = nodePositions;
        canvas._steps = steps;
    }

    // Canvas click → inspect node
    document.getElementById('pipeCanvas')?.addEventListener('click', (e) => {
        const canvas = e.target;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / pipeZoom;
        const y = (e.clientY - rect.top) / pipeZoom;
        const positions = canvas._nodePositions || {};
        const steps = canvas._steps || [];
        for (const s of steps) {
            const p = positions[s.id];
            if (p && x >= p.x && x <= p.x + NODE_W && y >= p.y && y <= p.y + NODE_H) {
                showStepDetail(s);
                return;
            }
        }
    });

    function showStepDetail(step) {
        const detail = document.getElementById('pipeStepDetail');
        const title = document.getElementById('pipeDetailTitle');
        if (!detail) return;
        if (title) title.textContent = step.name || step.id;
        const color = STEP_COLORS[step.type] || '#64748b';
        detail.innerHTML = `
            <div style="display:flex;flex-direction:column;gap:10px;font-size:13px">
                <div><span style="color:var(--text-3)">ID:</span> <span style="font-family:var(--font-mono)">${step.id}</span></div>
                <div><span style="color:var(--text-3)">Type:</span> <span style="color:${color};font-weight:600">${(step.type || 'tool_call').toUpperCase()}</span></div>
                ${step.description ? `<div><span style="color:var(--text-3)">Description:</span> ${step.description}</div>` : ''}
                ${step.tool_name ? `<div><span style="color:var(--text-3)">Tool:</span> <code style="background:var(--bg-3);padding:2px 6px;border-radius:4px">${step.tool_name}</code></div>` : ''}
                ${step.prompt_template ? `<div><span style="color:var(--text-3)">Prompt:</span> <span style="font-style:italic;color:var(--text-2)">${step.prompt_template.slice(0, 150)}${step.prompt_template.length > 150 ? '...' : ''}</span></div>` : ''}
                ${step.depends_on?.length ? `<div><span style="color:var(--text-3)">Depends on:</span> ${step.depends_on.join(', ')}</div>` : ''}
                ${step.status && step.status !== 'pending' ? `<div><span style="color:var(--text-3)">Status:</span> <span class="badge badge-${step.status === 'completed' ? 'allow' : step.status === 'failed' ? 'deny' : 'escalate'}">${step.status.toUpperCase()}</span></div>` : ''}
                ${step.duration_ms ? `<div><span style="color:var(--text-3)">Duration:</span> ${step.duration_ms.toFixed(0)}ms</div>` : ''}
                ${step.error ? `<div style="color:#dc2626"><span style="color:var(--text-3)">Error:</span> ${step.error}</div>` : ''}
                ${step.result ? `<div><span style="color:var(--text-3)">Result:</span> <pre style="background:var(--bg-3);padding:8px;border-radius:6px;font-size:11px;overflow:auto;max-height:120px">${JSON.stringify(step.result, null, 2)}</pre></div>` : ''}
                ${Object.keys(step.parameters || {}).length ? `<div><span style="color:var(--text-3)">Parameters:</span> <pre style="background:var(--bg-3);padding:8px;border-radius:6px;font-size:11px;overflow:auto;max-height:80px">${JSON.stringify(step.parameters, null, 2)}</pre></div>` : ''}
            </div>`;
    }

    // Zoom controls
    document.getElementById('pipeZoomIn')?.addEventListener('click', () => { pipeZoom = Math.min(pipeZoom + 0.15, 2.5); if (currentPipeline) renderPipelineDAG(currentPipeline); });
    document.getElementById('pipeZoomOut')?.addEventListener('click', () => { pipeZoom = Math.max(pipeZoom - 0.15, 0.3); if (currentPipeline) renderPipelineDAG(currentPipeline); });
    document.getElementById('pipeZoomFit')?.addEventListener('click', () => { pipeZoom = 1; pipeOffsetX = 0; pipeOffsetY = 0; if (currentPipeline) renderPipelineDAG(currentPipeline); });

    // Pan
    const pipeWrap = document.getElementById('pipeCanvasWrap');
    pipeWrap?.addEventListener('mousedown', (e) => { pipeDragging = true; pipeLastX = e.clientX; pipeLastY = e.clientY; pipeWrap.style.cursor = 'grabbing'; });
    window.addEventListener('mousemove', (e) => {
        if (!pipeDragging) return;
        pipeOffsetX += (e.clientX - pipeLastX) / pipeZoom;
        pipeOffsetY += (e.clientY - pipeLastY) / pipeZoom;
        pipeLastX = e.clientX; pipeLastY = e.clientY;
        if (currentPipeline) renderPipelineDAG(currentPipeline);
    });
    window.addEventListener('mouseup', () => { pipeDragging = false; if (pipeWrap) pipeWrap.style.cursor = 'grab'; });

    // Run from canvas
    document.getElementById('pipeRunBtn')?.addEventListener('click', () => {
        if (currentPipeline?.id) window._pipeRun(currentPipeline.id);
        else toast('No pipeline loaded', 'error');
    });
    document.getElementById('pipeNewBtn')?.addEventListener('click', () => {
        currentPipeline = null;
        document.getElementById('pipeDescription').value = '';
        document.getElementById('pipeEmptyPlaceholder').style.display = '';
        document.getElementById('pipeInfo').style.display = 'none';
        document.getElementById('pipeStepDetail').innerHTML = '<p class="muted">Click a node in the DAG to inspect it.</p>';
        document.getElementById('pipeDetailTitle').textContent = 'Step Details';
        document.getElementById('pipeCanvasTitle').textContent = 'Pipeline DAG';
        const ctx = document.getElementById('pipeCanvas')?.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, 9999, 9999);
    });
    document.getElementById('pipeRefreshBtn')?.addEventListener('click', loadPipelines);

    // ── Real-Time Pipeline Monitoring ──────────────────────────
    let pipeRunHistory = [];

    function handlePipelineStepUpdate(payload) {
        // Live update the DAG node status color
        if (currentPipeline && currentPipeline.steps) {
            const step = currentPipeline.steps.find(s => s.id === payload.step_id);
            if (step) {
                step.status = payload.status;
                step.duration_ms = payload.duration_ms;
                step.result = payload.result;
                step.error = payload.error;
                renderPipelineDAG(currentPipeline);
                addPipeChatMsg('ai', `\u2022 Step **${step.name || step.id}**: ${payload.status}${payload.duration_ms ? ` (${payload.duration_ms.toFixed(0)}ms)` : ''}`);
            }
        }
    }

    function handlePipelineComplete(payload) {
        const pipeName = payload.pipeline_name || payload.pipeline_id || 'Pipeline';
        const status = payload.status || 'completed';
        const msg = status === 'completed'
            ? `\u2705 **${pipeName}** completed! ${payload.steps_completed}/${payload.total_steps} steps in ${payload.duration_ms?.toFixed(0) || '?'}ms`
            : `\u274c **${pipeName}** failed at step ${payload.failed_step || '?'}: ${payload.error || 'Unknown error'}`;
        addPipeChatMsg('ai', msg);
        toast(msg.replace(/\*\*/g, ''), status === 'completed' ? 'success' : 'error');
        // Track run history
        pipeRunHistory.unshift({
            id: payload.pipeline_id, name: pipeName, status,
            steps: `${payload.steps_completed}/${payload.total_steps}`,
            duration: payload.duration_ms, ts: new Date().toLocaleTimeString('en-US', { hour12: false })
        });
        if (pipeRunHistory.length > 20) pipeRunHistory.length = 20;
        renderRunHistory();
        loadPipelines();
    }

    function renderRunHistory() {
        const body = document.getElementById('pipeHistoryBody');
        if (!body) return;
        if (!pipeRunHistory.length) {
            body.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-3)">No runs recorded this session</td></tr>';
            return;
        }
        body.innerHTML = pipeRunHistory.map(r => {
            const cls = r.status === 'completed' ? 'allow' : r.status === 'failed' ? 'deny' : 'escalate';
            return `<tr>
                <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-3)">${r.ts}</td>
                <td style="font-weight:600;color:var(--text-0)">${r.name}</td>
                <td>${r.steps}</td>
                <td><span class="badge badge-${cls}">${r.status.toUpperCase()}</span></td>
                <td style="font-family:var(--font-mono)">${r.duration ? r.duration.toFixed(0) + 'ms' : '\u2014'}</td>
            </tr>`;
        }).join('');
    }

    // \u2500\u2500 CIBIL Pipeline Health \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    async function loadPipelineHealth() {
        const data = await api('/pipelines/stats');
        if (!data) return;
        const totals = data.totals || {};
        const pipelines = data.pipelines || [];
        setText('#phSuccess .metric-num', totals.success_rate ? totals.success_rate + '%' : '\u2014');
        setText('#phTotal .metric-num', totals.total_runs || 0);
        setText('#phAvgTime .metric-num', totals.avg_duration_ms ? totals.avg_duration_ms.toFixed(0) + 'ms' : '\u2014');
        setText('#phActive .metric-num', totals.total_pipelines || 0);
        const grid = document.getElementById('pipeHealthGrid');
        if (!grid) return;
        if (!pipelines.length) { grid.innerHTML = '<p class="muted">Run pipelines to see health analytics.</p>'; return; }
        grid.innerHTML = pipelines.map(p => {
            const rateColor = p.success_rate >= 90 ? '#22c55e' : p.success_rate >= 70 ? '#f59e0b' : '#dc2626';
            const statusCls = p.last_status === 'completed' ? 'allow' : p.last_status === 'failed' ? 'deny' : 'proceed';
            const connectors = (p.connectors_used || []).map(c => `<span style="background:var(--bg-3);padding:2px 6px;border-radius:4px;font-size:10px;color:var(--text-2)">${c}</span>`).join(' ');
            return `<div class="card" style="padding:14px">
                <div style="font-weight:700;color:var(--text-0);margin-bottom:6px">${p.name}</div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                    <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden"><div style="width:${p.success_rate}%;height:100%;background:${rateColor};border-radius:3px"></div></div>
                    <span style="font-size:12px;font-weight:600;color:${rateColor}">${p.success_rate}%</span>
                </div>
                <div style="display:flex;gap:12px;font-size:11px;color:var(--text-3)">
                    <span>${p.runs} runs</span>
                    <span>${p.step_count} steps</span>
                    <span>${p.avg_duration_ms ? p.avg_duration_ms.toFixed(0) + 'ms avg' : 'No runs'}</span>
                </div>
                ${connectors ? `<div style="margin-top:6px;display:flex;gap:4px;flex-wrap:wrap">${connectors}</div>` : ''}
            </div>`;
        }).join('');
    }


    // ── Model Setup ───────────────────────────────────────────
    let modelEditingId = null;

    async function loadModels() {
        const data = await api('/models?include_disabled=true');
        const body = document.getElementById('modelTableBody');
        if (!body) return;
        const models = data?.models || [];
        // Stats
        const enabled = models.filter(m => m.enabled !== false);
        const providers = new Set(models.map(m => m.provider));
        setText('#mdlTotal .metric-num', models.length);
        setText('#mdlEnabled .metric-num', enabled.length);
        setText('#mdlProviders .metric-num', providers.size);
        // Table
        if (!models.length) {
            body.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-3)">No models configured. Use Quick Setup above!</td></tr>';
        } else {
            body.innerHTML = models.map(m => {
                const status = m.enabled !== false ? 'allow' : 'deny';
                const statusText = m.enabled !== false ? 'ENABLED' : 'DISABLED';
                const tasks = (m.task_categories || ['general']).join(', ');
                const isDefault = m.is_default ? ' <span style="color:#f59e0b;font-size:10px">\u2605 DEFAULT</span>' : '';
                return `<tr>
                    <td style="font-weight:600;color:var(--text-0)">${m.name || m.model_id}${isDefault}</td>
                    <td><span style="color:var(--text-2)">${m.provider || '?'}</span></td>
                    <td style="font-family:var(--font-mono);font-size:12px">${m.model_id}</td>
                    <td style="font-size:11px;color:var(--text-2);max-width:140px;overflow:hidden;text-overflow:ellipsis">${tasks}</td>
                    <td><span class="badge badge-${status}">${statusText}</span></td>
                    <td><span class="health-dot ${m._health || 'pending'}" id="mdlH_${m.model_id?.replace(/[^a-zA-Z0-9]/g, '_')}"></span></td>
                    <td>
                        <button class="text-btn" onclick="window._modelEdit('${m.model_id}')">Edit</button>
                        <button class="text-btn" onclick="window._modelToggle('${m.model_id}', ${m.enabled === false})">${m.enabled !== false ? 'Disable' : 'Enable'}</button>
                        <button class="text-btn" onclick="window._modelCheckHealth('${m.model_id}')">Check</button>
                        <button class="text-btn red" onclick="window._modelDelete('${m.model_id}')">Delete</button>
                    </td>
                </tr>`;
            }).join('');
        }
        // Task coverage
        renderTaskCoverage(models);
        // Fallback
        const stats = await api('/models/stats');
        if (stats?.fallback_chain?.length) {
            const chain = document.getElementById('modelFallbackChain');
            if (chain) chain.innerHTML = stats.fallback_chain.map((id, i) =>
                `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06)"><span style="background:var(--bg-3);width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--text-2)">${i + 1}</span><span style="font-family:var(--font-mono);font-size:12px">${id}</span></div>`
            ).join('');
        }
    }

    function renderTaskCoverage(models) {
        const el = document.getElementById('modelTaskCoverage');
        if (!el) return;
        const taskMap = {};
        const allTasks = ['general', 'pipeline_design', 'code_generation', 'data_analysis', 'conversation', 'security', 'summarization', 'fast'];
        allTasks.forEach(t => { taskMap[t] = []; });
        models.filter(m => m.enabled !== false).forEach(m => {
            (m.task_categories || ['general']).forEach(t => {
                if (!taskMap[t]) taskMap[t] = [];
                taskMap[t].push(m.name || m.model_id);
            });
        });
        el.innerHTML = allTasks.map(t => {
            const covered = taskMap[t]?.length > 0;
            const modelNames = taskMap[t]?.join(', ') || 'No model assigned';
            return `<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)"><span class="health-dot ${covered ? 'ok' : 'fail'}"></span><span style="font-size:12px;min-width:130px;color:var(--text-1)">${t}</span><span style="font-size:11px;color:var(--text-3)">${modelNames}</span></div>`;
        }).join('');
    }

    // Quick Setup
    window._modelQuickSetup = async function (provider) {
        const keyMap = { openai: 'qsKeyOpenai', anthropic: 'qsKeyAnthropic', gemini: 'qsKeyGemini', ollama: 'qsKeyOllama' };
        const keyEl = document.getElementById(keyMap[provider]);
        const key = keyEl?.value;
        if (!key) { toast('Enter an API key first', 'error'); return; }
        toast(`Setting up ${provider}...`, 'info');
        const endpoint = provider === 'ollama' ? '/models/quick-setup/ollama' : `/models/quick-setup/${provider}`;
        const body = provider === 'ollama' ? { api_key: key, model: 'llama3' } : { api_key: key };
        const data = await api(endpoint, { method: 'POST', body: JSON.stringify(body) });
        if (data) {
            toast(`${provider} configured successfully!`, 'success');
            keyEl.value = '';
            await loadModels();
        } else {
            toast(`Failed to setup ${provider}`, 'error');
        }
    };

    // Add/Edit modal
    document.getElementById('modelAddBtn')?.addEventListener('click', () => {
        modelEditingId = null;
        document.getElementById('modelModalTitle').textContent = 'Add Model';
        ['mdlName', 'mdlModelId', 'mdlApiKey', 'mdlBaseUrl'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
        document.getElementById('mdlProvider').value = 'openai';
        document.getElementById('mdlTasks').value = 'general';
        document.getElementById('mdlPriority').value = '0';
        document.getElementById('mdlMaxTokens').value = '4096';
        document.getElementById('mdlTemp').value = '0.7';
        document.getElementById('mdlDefault').checked = false;
        document.getElementById('modelModal').style.display = '';
    });
    document.getElementById('modelModalClose')?.addEventListener('click', () => document.getElementById('modelModal').style.display = 'none');
    document.getElementById('modelCancelBtn')?.addEventListener('click', () => document.getElementById('modelModal').style.display = 'none');

    document.getElementById('modelSaveBtn')?.addEventListener('click', async () => {
        const payload = {
            name: document.getElementById('mdlName')?.value || '',
            provider: document.getElementById('mdlProvider')?.value || 'openai',
            model_id: document.getElementById('mdlModelId')?.value || '',
            api_key: document.getElementById('mdlApiKey')?.value || '',
            base_url: document.getElementById('mdlBaseUrl')?.value || null,
            task_categories: (document.getElementById('mdlTasks')?.value || 'general').split(',').map(s => s.trim()).filter(Boolean),
            priority: parseInt(document.getElementById('mdlPriority')?.value || '0'),
            max_tokens: parseInt(document.getElementById('mdlMaxTokens')?.value || '4096'),
            temperature: parseFloat(document.getElementById('mdlTemp')?.value || '0.7'),
            is_default: document.getElementById('mdlDefault')?.checked || false,
        };
        if (!payload.model_id) { toast('Model ID is required', 'error'); return; }
        if (modelEditingId) {
            await api('/models/' + encodeURIComponent(modelEditingId), { method: 'PUT', body: JSON.stringify(payload) });
            toast('Model updated', 'success');
        } else {
            await api('/models', { method: 'POST', body: JSON.stringify(payload) });
            toast('Model added', 'success');
        }
        document.getElementById('modelModal').style.display = 'none';
        await loadModels();
    });

    window._modelEdit = async function (modelId) {
        const data = await api('/models/' + encodeURIComponent(modelId));
        if (!data) return;
        modelEditingId = modelId;
        document.getElementById('modelModalTitle').textContent = 'Edit Model: ' + modelId;
        document.getElementById('mdlName').value = data.name || '';
        document.getElementById('mdlProvider').value = data.provider || 'openai';
        document.getElementById('mdlModelId').value = data.model_id || '';
        document.getElementById('mdlApiKey').value = '';
        document.getElementById('mdlBaseUrl').value = data.base_url || '';
        document.getElementById('mdlTasks').value = (data.task_categories || ['general']).join(', ');
        document.getElementById('mdlPriority').value = data.priority || 0;
        document.getElementById('mdlMaxTokens').value = data.max_tokens || 4096;
        document.getElementById('mdlTemp').value = data.temperature ?? 0.7;
        document.getElementById('mdlDefault').checked = data.is_default || false;
        document.getElementById('modelModal').style.display = '';
    };

    window._modelToggle = async function (modelId, enable) {
        await api('/models/' + encodeURIComponent(modelId), { method: 'PUT', body: JSON.stringify({ enabled: enable }) });
        toast(enable ? 'Model enabled' : 'Model disabled', 'info');
        await loadModels();
    };

    window._modelDelete = async function (modelId) {
        await api('/models/' + encodeURIComponent(modelId), { method: 'DELETE' });
        toast('Model deleted', 'info');
        await loadModels();
    };

    window._modelCheckHealth = async function (modelId) {
        const dot = document.getElementById('mdlH_' + modelId.replace(/[^a-zA-Z0-9]/g, '_'));
        if (dot) dot.className = 'health-dot pending';
        const data = await api('/models/' + encodeURIComponent(modelId) + '/health');
        if (dot) dot.className = 'health-dot ' + (data?.healthy ? 'ok' : 'fail');
    };

    document.getElementById('modelHealthAllBtn')?.addEventListener('click', async () => {
        toast('Running health checks...', 'info');
        const data = await api('/models/health');
        if (data?.results) {
            let healthy = 0;
            data.results.forEach(r => {
                const dot = document.getElementById('mdlH_' + (r.model_id || '').replace(/[^a-zA-Z0-9]/g, '_'));
                if (dot) dot.className = 'health-dot ' + (r.healthy ? 'ok' : 'fail');
                if (r.healthy) healthy++;
            });
            setText('#mdlHealthy .metric-num', healthy);
            toast(`${healthy}/${data.results.length} models healthy`, healthy === data.results.length ? 'success' : 'error');
        }
    });

    document.getElementById('modelRefreshBtn')?.addEventListener('click', loadModels);

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
