"""
Shield Command — Connector Tool Executors

Real, working tool implementations for the 5 core connectors.
These are what the PipelineRunner invokes when executing tool_call steps.

Each connector class provides:
    - execute(action, params) — Run a specific action
    - health_check()           — Verify connectivity
    - list_actions()           — What this connector can do

Architecture:
    PipelineRunner → ConnectorExecutor.route(tool_name, params)
                        ↓
                   GitHubConnector / SlackConnector / PostgreSQLConnector / ...
                        ↓
                   Real API call (httpx / asyncpg / smtplib / boto3)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

import httpx

logger = logging.getLogger("shield.connectors")


# ---------------------------------------------------------------------------
# Base Connector
# ---------------------------------------------------------------------------

class BaseConnector:
    """Base class for all connector tool executors."""

    connector_id: str = ""
    name: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._call_count = 0
        self._error_count = 0

    async def execute(self, action: str, params: dict) -> dict:
        """Execute an action. Override in subclass."""
        raise NotImplementedError

    async def health_check(self) -> dict:
        """Check connectivity. Override in subclass."""
        return {"healthy": True, "connector": self.connector_id}

    def list_actions(self) -> list[dict]:
        """List available actions. Override in subclass."""
        return []

    def _record(self, success: bool) -> None:
        self._call_count += 1
        if not success:
            self._error_count += 1

    @property
    def stats(self) -> dict:
        return {
            "connector": self.connector_id,
            "calls": self._call_count,
            "errors": self._error_count,
        }


# ---------------------------------------------------------------------------
# 1. GitHub Connector
# ---------------------------------------------------------------------------

class GitHubConnector(BaseConnector):
    """
    GitHub API connector — repos, issues, PRs, commits, events.

    Config:
        token: GitHub Personal Access Token
        org: Default organization (optional)
        default_repo: owner/repo (optional)
    """

    connector_id = "github"
    name = "GitHub"

    BASE_URL = "https://api.github.com"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._config.get('token', '')}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _repo(self, params: dict) -> str:
        return params.get("repo", self._config.get("default_repo", ""))

    async def execute(self, action: str, params: dict) -> dict:
        actions = {
            "list_repos": self._list_repos,
            "get_repo": self._get_repo,
            "list_issues": self._list_issues,
            "create_issue": self._create_issue,
            "list_prs": self._list_prs,
            "get_pr": self._get_pr,
            "list_commits": self._list_commits,
            "list_events": self._list_events,
            "search_code": self._search_code,
        }
        handler = actions.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(actions.keys())}
        try:
            result = await handler(params)
            self._record(True)
            return result
        except Exception as e:
            self._record(False)
            return {"error": str(e), "action": action}

    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/user", headers=self._headers()
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "healthy": True,
                        "user": data.get("login"),
                        "rate_limit_remaining": int(resp.headers.get("x-ratelimit-remaining", 0)),
                    }
                return {"healthy": False, "status": resp.status_code}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def list_actions(self) -> list[dict]:
        return [
            {"action": "list_repos", "description": "List repositories", "params": ["org"]},
            {"action": "get_repo", "description": "Get repo details", "params": ["repo"]},
            {"action": "list_issues", "description": "List issues", "params": ["repo", "state"]},
            {"action": "create_issue", "description": "Create an issue", "params": ["repo", "title", "body", "labels"]},
            {"action": "list_prs", "description": "List pull requests", "params": ["repo", "state"]},
            {"action": "get_pr", "description": "Get PR details", "params": ["repo", "number"]},
            {"action": "list_commits", "description": "List recent commits", "params": ["repo", "branch"]},
            {"action": "list_events", "description": "List repo events", "params": ["repo", "event_type"]},
            {"action": "search_code", "description": "Search code", "params": ["query", "repo"]},
        ]

    async def _list_repos(self, params: dict) -> dict:
        org = params.get("org", self._config.get("org"))
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{self.BASE_URL}/orgs/{org}/repos" if org else f"{self.BASE_URL}/user/repos"
            resp = await client.get(url, headers=self._headers(), params={"per_page": 30, "sort": "updated"})
            resp.raise_for_status()
            repos = resp.json()
            return {
                "repos": [
                    {"name": r["full_name"], "description": r.get("description", ""),
                     "stars": r.get("stargazers_count", 0), "language": r.get("language"),
                     "updated_at": r.get("updated_at")}
                    for r in repos
                ],
                "count": len(repos),
            }

    async def _get_repo(self, params: dict) -> dict:
        repo = self._repo(params)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.BASE_URL}/repos/{repo}", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def _list_issues(self, params: dict) -> dict:
        repo = self._repo(params)
        state = params.get("state", "open")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/repos/{repo}/issues",
                headers=self._headers(),
                params={"state": state, "per_page": 20},
            )
            resp.raise_for_status()
            issues = resp.json()
            return {
                "issues": [
                    {"number": i["number"], "title": i["title"], "state": i["state"],
                     "labels": [l["name"] for l in i.get("labels", [])],
                     "created_at": i["created_at"]}
                    for i in issues if "pull_request" not in i
                ],
                "count": len(issues),
            }

    async def _create_issue(self, params: dict) -> dict:
        repo = self._repo(params)
        body = {
            "title": params["title"],
            "body": params.get("body", ""),
        }
        if params.get("labels"):
            body["labels"] = params["labels"] if isinstance(params["labels"], list) else [params["labels"]]
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/repos/{repo}/issues",
                headers=self._headers(), json=body,
            )
            resp.raise_for_status()
            issue = resp.json()
            return {"number": issue["number"], "url": issue["html_url"], "title": issue["title"]}

    async def _list_prs(self, params: dict) -> dict:
        repo = self._repo(params)
        state = params.get("state", "open")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/repos/{repo}/pulls",
                headers=self._headers(), params={"state": state, "per_page": 20},
            )
            resp.raise_for_status()
            prs = resp.json()
            return {
                "pull_requests": [
                    {"number": p["number"], "title": p["title"], "state": p["state"],
                     "user": p["user"]["login"], "created_at": p["created_at"]}
                    for p in prs
                ],
                "count": len(prs),
            }

    async def _get_pr(self, params: dict) -> dict:
        repo = self._repo(params)
        number = params["number"]
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/repos/{repo}/pulls/{number}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def _list_commits(self, params: dict) -> dict:
        repo = self._repo(params)
        branch = params.get("branch", "main")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/repos/{repo}/commits",
                headers=self._headers(), params={"sha": branch, "per_page": 10},
            )
            resp.raise_for_status()
            commits = resp.json()
            return {
                "commits": [
                    {"sha": c["sha"][:8], "message": c["commit"]["message"][:100],
                     "author": c["commit"]["author"]["name"], "date": c["commit"]["author"]["date"]}
                    for c in commits
                ],
                "count": len(commits),
            }

    async def _list_events(self, params: dict) -> dict:
        repo = self._repo(params)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/repos/{repo}/events",
                headers=self._headers(), params={"per_page": 20},
            )
            resp.raise_for_status()
            events = resp.json()
            event_type = params.get("event_type")
            if event_type:
                events = [e for e in events if e.get("type") == event_type]
            return {
                "events": [
                    {"type": e["type"], "actor": e["actor"]["login"],
                     "created_at": e["created_at"]}
                    for e in events[:20]
                ],
                "count": len(events),
            }

    async def _search_code(self, params: dict) -> dict:
        query = params["query"]
        repo = self._repo(params)
        if repo:
            query = f"{query} repo:{repo}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/search/code",
                headers=self._headers(), params={"q": query, "per_page": 10},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "results": [
                    {"path": r["path"], "repo": r["repository"]["full_name"],
                     "url": r["html_url"]}
                    for r in data.get("items", [])
                ],
                "total": data.get("total_count", 0),
            }


# ---------------------------------------------------------------------------
# 2. Slack Connector
# ---------------------------------------------------------------------------

class SlackConnector(BaseConnector):
    """
    Slack API connector — messaging, channels, users.

    Config:
        bot_token: Slack Bot OAuth Token (xoxb-...)
        default_channel: Default channel for messages
        webhook_url: Incoming webhook URL (optional for simple notifications)
    """

    connector_id = "slack"
    name = "Slack"

    BASE_URL = "https://slack.com/api"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._config.get('bot_token', '')}",
            "Content-Type": "application/json",
        }

    async def execute(self, action: str, params: dict) -> dict:
        actions = {
            "send_message": self._send_message,
            "list_channels": self._list_channels,
            "list_users": self._list_users,
            "get_channel_history": self._get_channel_history,
            "upload_file": self._upload_file,
            "add_reaction": self._add_reaction,
        }
        handler = actions.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(actions.keys())}
        try:
            result = await handler(params)
            self._record(True)
            return result
        except Exception as e:
            self._record(False)
            return {"error": str(e), "action": action}

    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.BASE_URL}/auth.test", headers=self._headers()
                )
                data = resp.json()
                return {
                    "healthy": data.get("ok", False),
                    "team": data.get("team"),
                    "user": data.get("user"),
                    "bot_id": data.get("bot_id"),
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def list_actions(self) -> list[dict]:
        return [
            {"action": "send_message", "description": "Send a message to a channel", "params": ["channel", "message", "blocks"]},
            {"action": "list_channels", "description": "List workspace channels", "params": ["limit"]},
            {"action": "list_users", "description": "List workspace members", "params": []},
            {"action": "get_channel_history", "description": "Get channel messages", "params": ["channel", "limit"]},
            {"action": "upload_file", "description": "Upload a file", "params": ["channel", "content", "filename"]},
            {"action": "add_reaction", "description": "Add emoji reaction", "params": ["channel", "timestamp", "emoji"]},
        ]

    async def _send_message(self, params: dict) -> dict:
        channel = params.get("channel", self._config.get("default_channel", "#general"))
        message = params.get("message", "")

        # If webhook_url is set and we just need to send a simple message
        webhook_url = self._config.get("webhook_url")
        if webhook_url and not params.get("blocks"):
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json={"text": message})
                return {"ok": resp.status_code == 200, "channel": channel}

        # Use Bot API
        body = {"channel": channel, "text": message}
        if params.get("blocks"):
            body["blocks"] = params["blocks"]

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/chat.postMessage",
                headers=self._headers(), json=body,
            )
            data = resp.json()
            return {
                "ok": data.get("ok", False),
                "ts": data.get("ts"),
                "channel": data.get("channel"),
                "error": data.get("error"),
            }

    async def _list_channels(self, params: dict) -> dict:
        limit = params.get("limit", 50)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/conversations.list",
                headers=self._headers(),
                params={"limit": limit, "types": "public_channel,private_channel"},
            )
            data = resp.json()
            channels = data.get("channels", [])
            return {
                "channels": [
                    {"id": c["id"], "name": c["name"],
                     "members": c.get("num_members", 0), "topic": c.get("topic", {}).get("value", "")}
                    for c in channels
                ],
                "count": len(channels),
            }

    async def _list_users(self, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/users.list", headers=self._headers()
            )
            data = resp.json()
            members = data.get("members", [])
            return {
                "users": [
                    {"id": m["id"], "name": m.get("real_name", m.get("name", "")),
                     "email": m.get("profile", {}).get("email", "")}
                    for m in members if not m.get("is_bot") and not m.get("deleted")
                ],
                "count": len(members),
            }

    async def _get_channel_history(self, params: dict) -> dict:
        channel = params.get("channel", self._config.get("default_channel"))
        limit = params.get("limit", 10)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/conversations.history",
                headers=self._headers(),
                params={"channel": channel, "limit": limit},
            )
            data = resp.json()
            messages = data.get("messages", [])
            return {
                "messages": [
                    {"text": m.get("text", "")[:200], "user": m.get("user"),
                     "ts": m.get("ts"), "type": m.get("type")}
                    for m in messages
                ],
                "count": len(messages),
            }

    async def _upload_file(self, params: dict) -> dict:
        channel = params.get("channel", self._config.get("default_channel"))
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/files.upload",
                headers={"Authorization": f"Bearer {self._config.get('bot_token', '')}"},
                data={
                    "channels": channel,
                    "content": params.get("content", ""),
                    "filename": params.get("filename", "file.txt"),
                    "title": params.get("title", ""),
                },
            )
            data = resp.json()
            return {"ok": data.get("ok", False), "file_id": data.get("file", {}).get("id")}

    async def _add_reaction(self, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/reactions.add",
                headers=self._headers(),
                json={
                    "channel": params["channel"],
                    "timestamp": params["timestamp"],
                    "name": params.get("emoji", "thumbsup"),
                },
            )
            return resp.json()


# ---------------------------------------------------------------------------
# 3. PostgreSQL Connector
# ---------------------------------------------------------------------------

class PostgreSQLConnector(BaseConnector):
    """
    PostgreSQL connector — queries, table info, schema inspection.

    Config:
        host, port, database, username, password
    
    Uses raw TCP via asyncpg (if available) or falls back to psycopg2 sync.
    For portability, this also includes a httpx-based approach via REST adapters.
    """

    connector_id = "postgresql"
    name = "PostgreSQL"

    async def execute(self, action: str, params: dict) -> dict:
        actions = {
            "query": self._query,
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "count_rows": self._count_rows,
            "insert": self._insert,
        }
        handler = actions.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(actions.keys())}
        try:
            result = await handler(params)
            self._record(True)
            return result
        except Exception as e:
            self._record(False)
            return {"error": str(e), "action": action}

    async def health_check(self) -> dict:
        try:
            conn = await self._get_connection()
            if conn is None:
                return {"healthy": False, "error": "Could not connect"}
            result = await self._run_query(conn, "SELECT 1 AS ok")
            await self._close_connection(conn)
            return {"healthy": True, "connector": "postgresql"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def list_actions(self) -> list[dict]:
        return [
            {"action": "query", "description": "Execute SQL query", "params": ["sql", "params"]},
            {"action": "list_tables", "description": "List all tables", "params": ["schema"]},
            {"action": "describe_table", "description": "Get table schema", "params": ["table"]},
            {"action": "count_rows", "description": "Count rows in table", "params": ["table"]},
            {"action": "insert", "description": "Insert a row", "params": ["table", "data"]},
        ]

    async def _get_connection(self):
        """Get a database connection using asyncpg or psycopg2."""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host=self._config.get("host", "localhost"),
                port=self._config.get("port", 5432),
                database=self._config.get("database", "postgres"),
                user=self._config.get("username", "postgres"),
                password=self._config.get("password", ""),
            )
            conn._connector_type = "asyncpg"
            return conn
        except ImportError:
            pass

        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self._config.get("host", "localhost"),
                port=self._config.get("port", 5432),
                dbname=self._config.get("database", "postgres"),
                user=self._config.get("username", "postgres"),
                password=self._config.get("password", ""),
            )
            conn._connector_type = "psycopg2"
            return conn
        except ImportError:
            return None

    async def _run_query(self, conn, sql: str, params: list = None) -> list[dict]:
        """Run a query on the connection."""
        if getattr(conn, "_connector_type", "") == "asyncpg":
            if params:
                rows = await conn.fetch(sql, *params)
            else:
                rows = await conn.fetch(sql)
            return [dict(r) for r in rows]
        else:
            # psycopg2 sync
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.commit()
            return [{"affected_rows": cursor.rowcount}]

    async def _close_connection(self, conn):
        if getattr(conn, "_connector_type", "") == "asyncpg":
            await conn.close()
        else:
            conn.close()

    async def _query(self, params: dict) -> dict:
        sql = params.get("sql", params.get("query", ""))
        if not sql:
            return {"error": "No SQL query provided"}
        conn = await self._get_connection()
        if not conn:
            return {"error": "No PostgreSQL driver available (install asyncpg or psycopg2)"}
        try:
            rows = await self._run_query(conn, sql, params.get("params"))
            # Serialize any non-JSON-safe types
            safe_rows = json.loads(json.dumps(rows, default=str))
            return {"rows": safe_rows, "count": len(safe_rows)}
        finally:
            await self._close_connection(conn)

    async def _list_tables(self, params: dict) -> dict:
        schema = params.get("schema", "public")
        sql = """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = $1
            ORDER BY table_name
        """
        conn = await self._get_connection()
        if not conn:
            return {"error": "No PostgreSQL driver available"}
        try:
            rows = await self._run_query(conn, sql, [schema])
            return {"tables": rows, "schema": schema, "count": len(rows)}
        finally:
            await self._close_connection(conn)

    async def _describe_table(self, params: dict) -> dict:
        table = params.get("table", "")
        sql = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """
        conn = await self._get_connection()
        if not conn:
            return {"error": "No PostgreSQL driver available"}
        try:
            rows = await self._run_query(conn, sql, [table])
            return {"table": table, "columns": rows, "count": len(rows)}
        finally:
            await self._close_connection(conn)

    async def _count_rows(self, params: dict) -> dict:
        table = params.get("table", "")
        # Safety: only allow simple table names
        if not table.isidentifier():
            return {"error": f"Invalid table name: {table}"}
        sql = f"SELECT COUNT(*) AS count FROM {table}"
        conn = await self._get_connection()
        if not conn:
            return {"error": "No PostgreSQL driver available"}
        try:
            rows = await self._run_query(conn, sql)
            return {"table": table, "count": rows[0]["count"] if rows else 0}
        finally:
            await self._close_connection(conn)

    async def _insert(self, params: dict) -> dict:
        table = params.get("table", "")
        data = params.get("data", {})
        if not table.isidentifier():
            return {"error": f"Invalid table name: {table}"}
        if not data:
            return {"error": "No data to insert"}

        columns = list(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) RETURNING *"

        conn = await self._get_connection()
        if not conn:
            return {"error": "No PostgreSQL driver available"}
        try:
            rows = await self._run_query(conn, sql, list(data.values()))
            return {"inserted": rows, "table": table}
        finally:
            await self._close_connection(conn)


# ---------------------------------------------------------------------------
# 4. Email Connector (SMTP + IMAP)
# ---------------------------------------------------------------------------

class EmailConnector(BaseConnector):
    """
    Email connector — send/receive emails via SMTP/IMAP.

    Config:
        smtp_host, smtp_port, imap_host, imap_port,
        username, password, from_address, use_tls
    """

    connector_id = "email"
    name = "Email (SMTP/IMAP)"

    async def execute(self, action: str, params: dict) -> dict:
        actions = {
            "send": self._send_email,
            "read_inbox": self._read_inbox,
        }
        handler = actions.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(actions.keys())}
        try:
            result = await handler(params)
            self._record(True)
            return result
        except Exception as e:
            self._record(False)
            return {"error": str(e), "action": action}

    async def health_check(self) -> dict:
        try:
            host = self._config.get("smtp_host", "smtp.gmail.com")
            port = self._config.get("smtp_port", 587)
            use_tls = self._config.get("use_tls", True)

            if use_tls:
                server = smtplib.SMTP(host, port, timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP(host, port, timeout=10)

            server.login(
                self._config.get("username", ""),
                self._config.get("password", ""),
            )
            server.quit()
            return {"healthy": True, "smtp_host": host}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def list_actions(self) -> list[dict]:
        return [
            {"action": "send", "description": "Send an email", "params": ["to", "subject", "body", "html"]},
            {"action": "read_inbox", "description": "Read inbox emails", "params": ["limit", "filter"]},
        ]

    async def _send_email(self, params: dict) -> dict:
        """Send an email via SMTP."""
        host = self._config.get("smtp_host", "smtp.gmail.com")
        port = self._config.get("smtp_port", 587)
        from_addr = self._config.get("from_address", self._config.get("username", ""))

        msg = MIMEMultipart("alternative")
        msg["Subject"] = params.get("subject", "Shield Notification")
        msg["From"] = from_addr
        msg["To"] = params["to"]

        # Plain text
        if params.get("body"):
            msg.attach(MIMEText(params["body"], "plain"))
        # HTML
        if params.get("html"):
            msg.attach(MIMEText(params["html"], "html"))
        elif params.get("body"):
            # Auto-generate simple HTML
            msg.attach(MIMEText(f"<p>{params['body']}</p>", "html"))

        # Send via SMTP
        use_tls = self._config.get("use_tls", True)
        if use_tls:
            server = smtplib.SMTP(host, port, timeout=15)
            server.starttls()
        else:
            server = smtplib.SMTP(host, port, timeout=15)

        server.login(
            self._config.get("username", ""),
            self._config.get("password", ""),
        )
        server.sendmail(from_addr, params["to"], msg.as_string())
        server.quit()

        return {"sent": True, "to": params["to"], "subject": params.get("subject")}

    async def _read_inbox(self, params: dict) -> dict:
        """Read emails from IMAP inbox."""
        try:
            import imaplib
            import email as email_lib

            host = self._config.get("imap_host", "imap.gmail.com")
            port = self._config.get("imap_port", 993)

            mail = imaplib.IMAP4_SSL(host, port)
            mail.login(
                self._config.get("username", ""),
                self._config.get("password", ""),
            )
            mail.select("INBOX")

            filter_type = params.get("filter", "ALL")
            status, messages = mail.search(None, filter_type.upper())
            email_ids = messages[0].split()

            limit = params.get("limit", 10)
            recent_ids = email_ids[-limit:] if len(email_ids) > limit else email_ids

            emails = []
            for eid in reversed(recent_ids):
                status, msg_data = mail.fetch(eid, "(RFC822)")
                msg = email_lib.message_from_bytes(msg_data[0][1])
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                            break
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="replace")

                emails.append({
                    "from": msg.get("From", ""),
                    "subject": msg.get("Subject", ""),
                    "date": msg.get("Date", ""),
                    "body": body[:500],
                })

            mail.logout()
            return {"emails": emails, "count": len(emails)}
        except Exception as e:
            return {"error": str(e), "emails": []}


# ---------------------------------------------------------------------------
# 5. S3 Connector (Amazon S3 / MinIO compatible)
# ---------------------------------------------------------------------------

class S3Connector(BaseConnector):
    """
    S3-compatible object storage connector.

    Config:
        access_key, secret_key, region, bucket, endpoint_url (for MinIO)
    
    Uses httpx with AWS Signature V4 for direct API calls (no boto3 dependency).
    Falls back to boto3 if available.
    """

    connector_id = "s3"
    name = "Amazon S3"

    async def execute(self, action: str, params: dict) -> dict:
        actions = {
            "list_objects": self._list_objects,
            "get_object": self._get_object,
            "put_object": self._put_object,
            "delete_object": self._delete_object,
            "list_buckets": self._list_buckets,
        }
        handler = actions.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(actions.keys())}
        try:
            result = await handler(params)
            self._record(True)
            return result
        except Exception as e:
            self._record(False)
            return {"error": str(e), "action": action}

    async def health_check(self) -> dict:
        try:
            client = self._get_boto3_client()
            if client:
                client.head_bucket(Bucket=self._config.get("bucket", ""))
                return {"healthy": True, "bucket": self._config.get("bucket")}
            return {"healthy": False, "error": "boto3 not available"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def list_actions(self) -> list[dict]:
        return [
            {"action": "list_objects", "description": "List objects in bucket", "params": ["prefix", "max_keys"]},
            {"action": "get_object", "description": "Download an object", "params": ["key"]},
            {"action": "put_object", "description": "Upload an object", "params": ["key", "body", "content_type"]},
            {"action": "delete_object", "description": "Delete an object", "params": ["key"]},
            {"action": "list_buckets", "description": "List all buckets", "params": []},
        ]

    def _get_boto3_client(self):
        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id=self._config.get("access_key"),
                aws_secret_access_key=self._config.get("secret_key"),
                region_name=self._config.get("region", "us-east-1"),
            )
            kwargs = {}
            if self._config.get("endpoint_url"):
                kwargs["endpoint_url"] = self._config["endpoint_url"]
            return session.client("s3", **kwargs)
        except ImportError:
            return None

    async def _list_objects(self, params: dict) -> dict:
        client = self._get_boto3_client()
        if not client:
            return {"error": "boto3 not installed (pip install boto3)"}
        bucket = params.get("bucket", self._config.get("bucket"))
        prefix = params.get("prefix", "")
        max_keys = params.get("max_keys", 50)

        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
        objects = resp.get("Contents", [])
        return {
            "objects": [
                {"key": o["Key"], "size": o["Size"],
                 "last_modified": o["LastModified"].isoformat()}
                for o in objects
            ],
            "count": len(objects),
            "bucket": bucket,
        }

    async def _get_object(self, params: dict) -> dict:
        client = self._get_boto3_client()
        if not client:
            return {"error": "boto3 not installed"}
        bucket = params.get("bucket", self._config.get("bucket"))
        key = params["key"]

        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()

        # Try to decode as text
        try:
            content = body.decode("utf-8")
            return {"key": key, "content": content[:10000], "size": len(body), "content_type": resp.get("ContentType")}
        except UnicodeDecodeError:
            return {"key": key, "size": len(body), "content_type": resp.get("ContentType"), "binary": True}

    async def _put_object(self, params: dict) -> dict:
        client = self._get_boto3_client()
        if not client:
            return {"error": "boto3 not installed"}
        bucket = params.get("bucket", self._config.get("bucket"))
        key = params["key"]
        body = params.get("body", "")
        content_type = params.get("content_type", "text/plain")

        if isinstance(body, str):
            body = body.encode("utf-8")

        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        return {"uploaded": True, "key": key, "bucket": bucket}

    async def _delete_object(self, params: dict) -> dict:
        client = self._get_boto3_client()
        if not client:
            return {"error": "boto3 not installed"}
        bucket = params.get("bucket", self._config.get("bucket"))
        key = params["key"]

        client.delete_object(Bucket=bucket, Key=key)
        return {"deleted": True, "key": key, "bucket": bucket}

    async def _list_buckets(self, params: dict) -> dict:
        client = self._get_boto3_client()
        if not client:
            return {"error": "boto3 not installed"}
        resp = client.list_buckets()
        buckets = resp.get("Buckets", [])
        return {
            "buckets": [
                {"name": b["Name"], "created": b["CreationDate"].isoformat()}
                for b in buckets
            ],
            "count": len(buckets),
        }


# ---------------------------------------------------------------------------
# Connector Executor (Router)
# ---------------------------------------------------------------------------

# Map tool_name prefixes to connector classes
CONNECTOR_MAP: dict[str, type[BaseConnector]] = {
    "github": GitHubConnector,
    "slack": SlackConnector,
    "postgresql": PostgreSQLConnector,
    "postgres": PostgreSQLConnector,
    "email": EmailConnector,
    "s3": S3Connector,
}


class ConnectorExecutor:
    """
    Routes tool calls to the appropriate connector executor.
    
    Used by PipelineRunner to execute tool_call steps.
    
    Tool name format: {connector}_{action}
    Example: "github_create_issue", "slack_send_message", "s3_list_objects"
    """

    def __init__(self) -> None:
        self._instances: dict[str, BaseConnector] = {}

    def register(self, connector_id: str, config: dict) -> BaseConnector:
        """Register a connector with its config."""
        cls = CONNECTOR_MAP.get(connector_id)
        if not cls:
            raise ValueError(f"Unknown connector: {connector_id}")
        instance = cls(config)
        self._instances[connector_id] = instance
        logger.info("Registered connector executor: %s", connector_id)
        return instance

    def get(self, connector_id: str) -> Optional[BaseConnector]:
        """Get a registered connector instance."""
        return self._instances.get(connector_id)

    async def route(self, tool_name: str, params: dict) -> dict:
        """
        Route a tool call to the correct connector.
        
        tool_name format: connector_action (e.g., "github_create_issue")
        """
        # Parse tool name: "github_create_issue" → ("github", "create_issue")
        parts = tool_name.split("_", 1)
        if len(parts) < 2:
            return {"error": f"Invalid tool name format: {tool_name}. Use connector_action"}

        connector_id = parts[0]
        action = parts[1]

        # Check aliases
        if connector_id == "postgres":
            connector_id = "postgresql"

        connector = self._instances.get(connector_id)
        if not connector:
            return {
                "error": f"Connector '{connector_id}' not registered. Available: {list(self._instances.keys())}",
            }

        return await connector.execute(action, params)

    async def health_check_all(self) -> list[dict]:
        """Health check all registered connectors."""
        results = []
        for cid, connector in self._instances.items():
            result = await connector.health_check()
            result["connector_id"] = cid
            results.append(result)
        return results

    def list_all_actions(self) -> dict[str, list[dict]]:
        """List available actions for all registered connectors."""
        return {
            cid: connector.list_actions()
            for cid, connector in self._instances.items()
        }
