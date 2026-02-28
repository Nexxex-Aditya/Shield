"""
Shield Command — Reactive Neural Bus

A pub/sub event bus where EVERY component streams events in real-time.
Not just token streaming — security decisions, memory activations,
graph mutations, tool calls, and diagnostics all flow through one bus.

The dashboard subscribes via WebSocket and gets a live "brain activity"
feed showing what the agent is thinking, doing, and deciding.

Integration points:
    - MCPGateway: emits security decisions
    - CognitiveGraph: emits node status / mutations
    - CognitiveMemory: emits memory reads / writes
    - PromptGuard: emits real-time injection scan results
    - All modules: emit typed events
"""

import asyncio
import time
import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable, Awaitable
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("shield.neural_bus")


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventChannel(str, Enum):
    """Named channels for different event types."""
    TOKENS = "tokens"                # LLM token stream
    SECURITY = "security"            # Policy decisions, blocks, alerts
    MEMORY = "memory"                # Memory reads, writes, consolidation
    GRAPH = "graph"                  # Node status, mutations, edges
    TOOLS = "tools"                  # Tool calls, results, errors
    DIAGNOSTICS = "diagnostics"      # Performance, cost, health
    RETRIEVAL = "retrieval"          # RAG search, reranking
    GENETICS = "genetics"            # Evolution events
    MESH = "mesh"                    # Agent collaboration events
    SYSTEM = "system"                # Startup, shutdown, errors
    DEPLOY = "deploy"                # Deployment pipeline events


@dataclass
class BusEvent:
    """A typed event flowing through the neural bus."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    channel: str = "system"
    event_type: str = ""           # e.g., "token_generated", "policy_blocked"
    source: str = ""               # e.g., "MCPGateway", "CognitiveGraph"
    data: Any = None
    agent_id: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(d.get("data"), (dict, list, str, int, float, bool, type(None))):
            pass  # Already serializable
        else:
            d["data"] = str(d["data"])
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------

@dataclass
class Subscription:
    """A subscription to one or more channels."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    channels: list[str] = field(default_factory=list)
    callback: Optional[Callable] = None  # async callback(BusEvent)
    queue: Optional[asyncio.Queue] = None  # or queue-based
    active: bool = True
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Neural Bus
# ---------------------------------------------------------------------------

class NeuralBus:
    """
    Central pub/sub event bus for real-time streaming.
    
    Usage:
        bus = NeuralBus()
        
        # Subscribe with callback
        sub_id = await bus.subscribe(
            channels=["tokens", "security"],
            callback=my_handler,
        )
        
        # Subscribe with queue (for WebSocket forwarding)
        queue = asyncio.Queue()
        sub_id = await bus.subscribe(
            channels=["*"],  # all channels
            queue=queue,
        )
        
        # Publish events (from any module)
        await bus.emit(BusEvent(
            channel="tokens",
            event_type="token_generated",
            source="LLM",
            data={"token": "Hello", "index": 0},
        ))
        
        # Stream tokens
        async for token in bus.token_stream("session-123"):
            print(token, end="", flush=True)
    """

    def __init__(self, max_history: int = 1000):
        self._subscriptions: dict[str, Subscription] = {}
        self._channel_subs: dict[str, list[str]] = defaultdict(list)  # channel → [sub_ids]
        self._history: list[BusEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

        # Stats
        self.total_events = 0
        self.events_per_channel: dict[str, int] = defaultdict(int)

    # ── Subscribe ─────────────────────────────────────────────────

    async def subscribe(
        self,
        channels: list[str] = None,
        callback: Callable = None,
        queue: asyncio.Queue = None,
    ) -> str:
        """
        Subscribe to events.
        
        Args:
            channels: list of channel names, or ["*"] for all
            callback: async function(BusEvent) called for each event
            queue: asyncio.Queue to push events into
        """
        sub = Subscription(
            channels=channels or ["*"],
            callback=callback,
            queue=queue,
        )

        async with self._lock:
            self._subscriptions[sub.id] = sub
            for ch in sub.channels:
                self._channel_subs[ch].append(sub.id)

        logger.debug(f"New subscription {sub.id}: channels={sub.channels}")
        return sub.id

    async def unsubscribe(self, sub_id: str):
        """Remove a subscription."""
        async with self._lock:
            sub = self._subscriptions.pop(sub_id, None)
            if sub:
                sub.active = False
                for ch in sub.channels:
                    if sub_id in self._channel_subs[ch]:
                        self._channel_subs[ch].remove(sub_id)

    # ── Publish ───────────────────────────────────────────────────

    async def emit(self, event: BusEvent):
        """Publish an event to all matching subscribers."""
        self.total_events += 1
        self.events_per_channel[event.channel] += 1

        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Find matching subscribers
        matching_sub_ids = set()
        matching_sub_ids.update(self._channel_subs.get(event.channel, []))
        matching_sub_ids.update(self._channel_subs.get("*", []))

        for sub_id in matching_sub_ids:
            sub = self._subscriptions.get(sub_id)
            if not sub or not sub.active:
                continue

            try:
                if sub.callback:
                    if asyncio.iscoroutinefunction(sub.callback):
                        await sub.callback(event)
                    else:
                        sub.callback(event)
                if sub.queue:
                    sub.queue.put_nowait(event)
            except Exception as e:
                logger.warning(f"Error delivering event to {sub_id}: {e}")

    # ── Convenience Emitters ──────────────────────────────────────

    async def emit_token(self, token: str, session_id: str = "", source: str = "llm"):
        """Emit a token event."""
        await self.emit(BusEvent(
            channel=EventChannel.TOKENS,
            event_type="token_generated",
            source=source,
            data={"token": token},
            session_id=session_id,
        ))

    async def emit_security(self, event_type: str, data: dict, source: str = "MCPGateway"):
        """Emit a security event."""
        await self.emit(BusEvent(
            channel=EventChannel.SECURITY,
            event_type=event_type,
            source=source,
            data=data,
        ))

    async def emit_tool_call(self, tool_name: str, params: dict, result: Any = None, source: str = ""):
        """Emit a tool call event."""
        await self.emit(BusEvent(
            channel=EventChannel.TOOLS,
            event_type="tool_executed",
            source=source,
            data={"tool": tool_name, "params": params, "result": result},
        ))

    async def emit_graph_mutation(self, mutation_type: str, node_id: str, data: dict = None):
        """Emit a graph mutation event."""
        await self.emit(BusEvent(
            channel=EventChannel.GRAPH,
            event_type=f"graph_{mutation_type}",
            source="CognitiveGraph",
            data={"node_id": node_id, **(data or {})},
        ))

    async def emit_memory(self, operation: str, data: dict):
        """Emit a memory event."""
        await self.emit(BusEvent(
            channel=EventChannel.MEMORY,
            event_type=f"memory_{operation}",
            source="CognitiveMemory",
            data=data,
        ))

    # ── Streaming ─────────────────────────────────────────────────

    async def token_stream(self, session_id: str = "") -> asyncio.Queue:
        """
        Create a queue that receives token events for a session.
        Use with WebSocket to stream tokens to the dashboard.
        """
        queue = asyncio.Queue()
        
        async def filter_tokens(event: BusEvent):
            if event.channel == EventChannel.TOKENS:
                if not session_id or event.session_id == session_id:
                    await queue.put(event.data.get("token", "") if isinstance(event.data, dict) else str(event.data))

        await self.subscribe(channels=[EventChannel.TOKENS], callback=filter_tokens)
        return queue

    # ── History & Stats ───────────────────────────────────────────

    def get_recent(self, channel: str = None, limit: int = 50) -> list[dict]:
        """Get recent events, optionally filtered by channel."""
        events = self._history
        if channel:
            events = [e for e in events if e.channel == channel]
        return [e.to_dict() for e in events[-limit:]]

    async def get_stats(self) -> dict:
        return {
            "total_events": self.total_events,
            "active_subscriptions": sum(1 for s in self._subscriptions.values() if s.active),
            "events_per_channel": dict(self.events_per_channel),
            "history_size": len(self._history),
        }
