# 🛡️ Shield Command

**Intelligent Middleware Platform for Autonomous Systems**

Shield Command is a production-grade operations automation platform that lets companies design, build, and run automated workflows with AI-powered pipeline compilation, visual DAG editing, real-time monitoring, and enterprise security.

---

## Quick Start

```bash
# 1. Initialize a project
python shield_cli.py init my-project
cd my-project

# 2. Add an AI model
python shield_cli.py models add openai sk-your-key-here

# 3. Start the server
python shield_cli.py server

# 4. Open the dashboard
# → http://localhost:8000
```

## Docker

```bash
# Build and deploy
docker-compose up --build -d

# Monitor
docker-compose logs -f shield

# Stop
docker-compose down
```

---

## Features

### Pipeline Engine
- **Natural Language → DAG**: Describe a workflow in plain English, get a compiled pipeline
- **5 Built-in Templates**: GitHub→Slack, Email→Jira, Data Report, Deploy+Monitor, Content Gen
- **Visual DAG Editor**: Canvas-based editor with zoom/pan, click-to-inspect, color-coded nodes
- **Conversational Design**: Chat-style interface to iteratively refine pipeline design
- **Real-Time Monitoring**: WebSocket-powered live status updates on running pipelines

### Connectors (27 Actions)
| Connector | Actions |
|-----------|---------|
| GitHub | repos, issues, PRs, commits, events, code search (9) |
| Slack | messages, channels, users, files, reactions (6) |
| PostgreSQL | queries, tables, schema, inserts (5) |
| Email | SMTP send, IMAP read (2) |
| S3 | object CRUD, buckets (5) |

### Model Management
- **Multi-Provider**: OpenAI, Anthropic, Google Gemini, Ollama (local)
- **Quick Setup**: One-click configuration for each provider
- **Encrypted Keys**: Fernet encryption for all API keys at rest
- **Task Routing**: Automatic model selection per task category
- **Fallback Chains**: Ordered fallback if primary model fails
- **Health Checks**: Probe all models with one click

### Security Pipeline
- 12-step security pipeline for every tool call
- Policy engine with YAML-defined rules
- Audit chain with tamper-proof logging
- Rate limiting and sandboxed execution
- Shadow execution for pre-flight safety checks
- Bidirectional surveillance monitoring
- CIBIL trust scoring for AI models

---

## CLI Reference

```
shield compile "description"     Compile natural language → pipeline DAG
shield run <id|file.yaml>        Execute a saved pipeline
shield templates                 List built-in pipeline templates
shield models [list|add|remove]  Manage AI model providers
shield health                    Health check all configured models
shield server                    Start the Shield API server
shield init [directory]          Scaffold a new Shield project
shield deploy [--build] [--detach]  Build & start with Docker
shield monitor [--interval N]    Live terminal monitoring dashboard
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/evaluate` | Evaluate and execute a tool call |
| GET | `/api/v1/audit` | Query audit log |
| GET/POST | `/api/v1/models` | List/add AI models |
| POST | `/api/v1/models/quick-setup/{provider}` | Quick setup for provider |
| GET | `/api/v1/models/{id}/health` | Health check a model |
| POST | `/api/v1/pipelines/compile` | Compile pipeline from description |
| POST | `/api/v1/pipelines/run` | Execute a pipeline |
| GET | `/api/v1/pipelines/stats` | Pipeline health analytics |
| GET | `/api/v1/pipelines/templates` | List built-in templates |
| WS | `/ws/live` | Real-time event stream |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Dashboard (Vanilla JS)               │
│  Overview · Pipelines · Models · CIBIL · Security     │
├──────────────────────────────────────────────────────┤
│                   FastAPI Server                      │
│  REST API (60+ endpoints) · WebSocket · Static Files  │
├──────────────────────────────────────────────────────┤
│                   Core Engine                         │
│  MCP Gateway · Policy Engine · Audit Chain           │
│  Pipeline Compiler/Runner · Model Registry           │
│  Connector Executor · Shadow Engine · CIBIL Scoring  │
├──────────────────────────────────────────────────────┤
│                   Data Layer                          │
│  SQLite/PostgreSQL · Fernet Encryption · YAML Store  │
└──────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- Dependencies: `pip install -r requirements.txt`

## License

MIT
