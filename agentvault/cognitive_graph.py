"""
Shield Command — Self-Modifying Cognitive Graph (SMCG)

The core agent runtime. Unlike static DAGs, this graph rewrites itself
during execution based on what the agent discovers.

Architecture:
    1. User provides a goal
    2. LLM decomposes it into an initial graph of sub-goals
    3. Each node executes through MCPGateway (security-governed)
    4. After each node, the LLM reflects and can MUTATE the graph:
       - SPAWN: add new nodes downstream
       - KILL: remove an entire branch
       - RETRY_ALT: replace a failed node with an alternative approach
       - MERGE: combine parallel results into a single node
    5. Every mutation is security-checked by MCPGateway

Integration points:
    - MCPGateway: every tool call goes through the 12-step security pipeline
    - ModelRegistry: routes LLM calls to the best available model
    - CognitiveMemory: informs decisions with past experience + learned rules
    - AuditChain: every mutation is logged for compliance
"""

import asyncio
import json
import uuid
import time
import logging
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable

logger = logging.getLogger("shield.cognitive_graph")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    WAITING = "waiting"  # waiting for human approval or sub-agent


class MutationType(str, Enum):
    SPAWN = "spawn"          # Add a new node after current
    KILL = "kill"            # Remove a branch
    RETRY_ALT = "retry_alt"  # Replace failed node with alternative
    MERGE = "merge"          # Combine parallel results


@dataclass
class GraphNode:
    """A single executable unit in the cognitive graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    tool_name: str = ""            # Tool to call (if tool-based)
    tool_params: dict = field(default_factory=dict)
    node_type: str = "action"      # action | decision | observation | synthesis
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: float = 0.0
    attempt: int = 0
    max_attempts: int = 3
    security_decision: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphMutation:
    """A proposed modification to the graph."""
    type: MutationType
    reason: str = ""
    target_node_id: Optional[str] = None  # For KILL/RETRY_ALT
    new_node: Optional[GraphNode] = None  # For SPAWN
    insert_after: Optional[str] = None    # Where to insert SPAWN
    approved: bool = False


@dataclass
class GraphExecutionResult:
    """Result of executing the entire cognitive graph."""
    goal: str
    status: str  # completed | failed | partial
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    killed_nodes: int = 0
    mutations_applied: int = 0
    mutations_blocked: int = 0
    duration_ms: float = 0.0
    final_output: Any = None
    node_trace: list[dict] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Cognitive Graph Runtime
# ---------------------------------------------------------------------------

class CognitiveGraph:
    """
    A self-modifying directed graph that serves as the agent's runtime.
    
    The graph starts from a goal, decomposes it using an LLM, executes
    nodes through the security gateway, and rewrites itself based on
    what it discovers during execution.
    """

    def __init__(
        self,
        gateway=None,           # MCPGateway instance
        model_registry=None,    # ModelRegistry instance
        memory=None,            # CognitiveMemoryManager instance
        audit=None,             # AuditChain instance
        max_nodes: int = 50,    # Safety limit
        max_mutations: int = 20,
        max_depth: int = 10,
    ):
        self.gateway = gateway
        self.model_registry = model_registry
        self.memory = memory
        self.audit = audit
        self.max_nodes = max_nodes
        self.max_mutations = max_mutations
        self.max_depth = max_depth
        
        self.nodes: dict[str, GraphNode] = {}
        self.root_ids: list[str] = []
        self.mutations_applied: int = 0
        self.mutations_blocked: int = 0
        self._listeners: list[Callable] = []

    # ── Graph Construction ────────────────────────────────────────

    def add_node(self, node: GraphNode, after: Optional[str] = None) -> str:
        """Add a node to the graph, optionally linking it after an existing node."""
        if len(self.nodes) >= self.max_nodes:
            raise RuntimeError(f"Graph node limit reached ({self.max_nodes})")
        
        self.nodes[node.id] = node
        
        if after and after in self.nodes:
            parent = self.nodes[after]
            parent.child_ids.append(node.id)
            node.parent_ids.append(after)
        elif not node.parent_ids:
            self.root_ids.append(node.id)
        
        self._emit("node_added", {"node_id": node.id, "goal": node.goal, "after": after})
        return node.id

    def remove_branch(self, node_id: str) -> list[str]:
        """Recursively remove a node and all its descendants."""
        removed = []
        self._remove_recursive(node_id, removed)
        self._emit("branch_removed", {"root": node_id, "removed": removed})
        return removed

    def _remove_recursive(self, node_id: str, removed: list[str]):
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        for child_id in list(node.child_ids):
            self._remove_recursive(child_id, removed)
        node.status = NodeStatus.KILLED
        removed.append(node_id)
        # Clean up parent references
        for pid in node.parent_ids:
            if pid in self.nodes:
                parent = self.nodes[pid]
                if node_id in parent.child_ids:
                    parent.child_ids.remove(node_id)
        if node_id in self.root_ids:
            self.root_ids.remove(node_id)

    def replace_node(self, old_id: str, new_node: GraphNode) -> str:
        """Replace a node with an alternative (keep connections)."""
        if old_id not in self.nodes:
            raise KeyError(f"Node {old_id} not found")
        old = self.nodes[old_id]
        new_node.parent_ids = list(old.parent_ids)
        new_node.child_ids = list(old.child_ids)
        # Update parent references
        for pid in new_node.parent_ids:
            if pid in self.nodes:
                parent = self.nodes[pid]
                parent.child_ids = [new_node.id if c == old_id else c for c in parent.child_ids]
        # Update child references
        for cid in new_node.child_ids:
            if cid in self.nodes:
                child = self.nodes[cid]
                child.parent_ids = [new_node.id if p == old_id else p for p in child.parent_ids]
        old.status = NodeStatus.KILLED
        self.nodes[new_node.id] = new_node
        if old_id in self.root_ids:
            self.root_ids = [new_node.id if r == old_id else r for r in self.root_ids]
        self._emit("node_replaced", {"old": old_id, "new": new_node.id})
        return new_node.id

    # ── Execution ─────────────────────────────────────────────────

    async def execute(self, goal: str, agent_id: str = "cognitive-agent", session_id: str = "") -> GraphExecutionResult:
        """
        Execute the full cognitive loop:
        1. Decompose the goal into an initial graph
        2. Execute nodes in topological order
        3. After each node, reflect and potentially mutate the graph
        4. Collect results and return
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())[:12]

        # Reset memory for this run
        if self.memory:
            self.memory.reset_working()
            self.memory.working.set("goal", goal)
            self.memory.working.set("agent_id", agent_id)

        try:
            # Phase 1: Decompose goal into initial graph
            await self._decompose_goal(goal, agent_id)
            self._emit("graph_initialized", {
                "goal": goal,
                "nodes": len(self.nodes),
                "roots": self.root_ids,
            })

            # Phase 2: Execute with self-modification
            await self._execute_loop(agent_id, session_id)

            # Phase 3: Synthesize final output
            final_output = await self._synthesize_results(goal, agent_id)

            # Record the experience
            if self.memory:
                completed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.COMPLETED)
                failed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED)
                emotional_tag = "success" if failed == 0 else "failure" if completed == 0 else "surprise"
                await self.memory.record_episode(
                    content=f"Goal: {goal}. Completed {completed}/{len(self.nodes)} nodes. Mutations: {self.mutations_applied}.",
                    tags=["cognitive_graph", agent_id],
                    impact_score=0.7,
                    emotional_tag=emotional_tag,
                    agent_id=agent_id,
                    run_id=session_id,
                )

            completed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.COMPLETED)
            failed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED)
            killed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.KILLED)

            status = "completed" if failed == 0 else "partial" if completed > 0 else "failed"

            return GraphExecutionResult(
                goal=goal,
                status=status,
                total_nodes=len(self.nodes),
                completed_nodes=completed,
                failed_nodes=failed,
                killed_nodes=killed,
                mutations_applied=self.mutations_applied,
                mutations_blocked=self.mutations_blocked,
                duration_ms=(time.time() - start_time) * 1000,
                final_output=final_output,
                node_trace=[self._node_to_trace(n) for n in self.nodes.values()],
            )

        except Exception as e:
            logger.error(f"Cognitive graph execution failed: {e}")
            return GraphExecutionResult(
                goal=goal,
                status="failed",
                total_nodes=len(self.nodes),
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _decompose_goal(self, goal: str, agent_id: str):
        """Use LLM to decompose a goal into sub-tasks (initial graph nodes)."""
        adapter = None
        if self.model_registry:
            adapter = self.model_registry.get_adapter("reasoning")
        
        if adapter:
            # Recall relevant memories
            memory_context = ""
            if self.memory:
                bundle = await self.memory.remember(goal, agent_filter=agent_id)
                if bundle.rules:
                    memory_context = "\n\nRelevant learned rules:\n" + "\n".join(
                        f"- {r.rule}" for r in bundle.rules[:5]
                    )
                if bundle.episodes:
                    memory_context += "\n\nRelevant past experiences:\n" + "\n".join(
                        f"- [{e.emotional_tag}] {e.content[:100]}" for e in bundle.episodes[:3]
                    )

            # Get available tools
            tools_desc = ""
            if self.gateway:
                tools = self.gateway.list_tools()
                if tools:
                    tools_desc = "\n\nAvailable tools:\n" + "\n".join(
                        f"- {t.get('name', 'unknown')}: {t.get('description', '')}" 
                        for t in tools[:20]
                    )

            try:
                response = await adapter.generate(
                    system_prompt=(
                        "You are a task decomposition engine. Given a goal, break it into "
                        "2-6 concrete sub-tasks. Each sub-task should be independently executable. "
                        "Respond with a JSON array of objects, each with 'goal' (string), "
                        "'tool_name' (string, or empty if no tool needed), "
                        "'tool_params' (object), and 'type' (one of: action, decision, observation, synthesis).\n"
                        "Only output valid JSON, nothing else."
                        + tools_desc + memory_context
                    ),
                    user_prompt=goal,
                    temperature=0.3,
                    max_tokens=1000,
                )
                
                content = response.get("content", "")
                # Parse JSON from response
                steps = self._parse_json_array(content)
                
                for i, step in enumerate(steps):
                    node = GraphNode(
                        goal=step.get("goal", f"Step {i+1}"),
                        tool_name=step.get("tool_name", ""),
                        tool_params=step.get("tool_params", {}),
                        node_type=step.get("type", "action"),
                    )
                    after_id = list(self.nodes.keys())[-1] if self.nodes else None
                    self.add_node(node, after=after_id)
                    
            except Exception as e:
                logger.warning(f"LLM decomposition failed ({e}), creating single-node graph")
                self.add_node(GraphNode(goal=goal, node_type="action"))
        else:
            # No LLM available — create a single node
            self.add_node(GraphNode(goal=goal, node_type="action"))

    async def _execute_loop(self, agent_id: str, session_id: str):
        """Execute nodes in topological order with self-modification."""
        iteration = 0
        max_iterations = self.max_nodes * 2  # Safety bound

        while iteration < max_iterations:
            iteration += 1
            
            # Find next executable nodes (all parents completed)
            executable = self._get_executable_nodes()
            if not executable:
                break
            
            # Execute all ready nodes (potentially in parallel)
            for node in executable:
                await self._execute_single_node(node, agent_id, session_id)
                
                # Reflect and potentially mutate
                if self.mutations_applied < self.max_mutations:
                    await self._reflect_and_mutate(node, agent_id)

        # Check for stuck nodes
        pending = [n for n in self.nodes.values() if n.status == NodeStatus.PENDING]
        if pending:
            logger.warning(f"{len(pending)} nodes stuck in PENDING state")

    async def _execute_single_node(self, node: GraphNode, agent_id: str, session_id: str):
        """Execute a single graph node."""
        node.status = NodeStatus.RUNNING
        node.started_at = time.time()
        node.attempt += 1
        self._emit("node_started", {"node_id": node.id, "goal": node.goal})

        try:
            if node.tool_name and self.gateway:
                # Execute through MCP Gateway (full security pipeline)
                result = await self.gateway.handle_tool_call(
                    agent_id=agent_id,
                    session_id=session_id,
                    tool_name=node.tool_name,
                    parameters=node.tool_params,
                    context={"goal": node.goal, "graph_node": node.id},
                )
                node.result = result.get("result") if isinstance(result, dict) else result
                node.security_decision = result.get("decision") if isinstance(result, dict) else "allow"
                
                if isinstance(result, dict) and result.get("decision") == "deny":
                    node.status = NodeStatus.FAILED
                    node.error = f"Security denied: {result.get('reason', 'policy violation')}"
                else:
                    node.status = NodeStatus.COMPLETED

            elif node.node_type == "synthesis":
                # Gather results from parent nodes
                parent_results = {
                    pid: self.nodes[pid].result
                    for pid in node.parent_ids
                    if pid in self.nodes and self.nodes[pid].result is not None
                }
                node.result = await self._synthesize_node(node, parent_results, agent_id)
                node.status = NodeStatus.COMPLETED

            else:
                # LLM-only reasoning node
                node.result = await self._reason_node(node, agent_id)
                node.status = NodeStatus.COMPLETED

        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            logger.error(f"Node {node.id} failed: {e}")

        node.completed_at = time.time()
        node.duration_ms = (node.completed_at - node.started_at) * 1000

        # Store in working memory
        if self.memory:
            self.memory.working.set(f"node_{node.id}_result", node.result)
            self.memory.working.set(f"node_{node.id}_status", node.status.value)

        self._emit("node_completed", {
            "node_id": node.id,
            "status": node.status.value,
            "duration_ms": node.duration_ms,
        })

    async def _reflect_and_mutate(self, node: GraphNode, agent_id: str):
        """After executing a node, decide if the graph should be modified."""
        adapter = None
        if self.model_registry:
            adapter = self.model_registry.get_adapter("reasoning")
        if not adapter:
            return

        # Build reflection context
        graph_state = {
            "just_completed": {"id": node.id, "goal": node.goal, "status": node.status.value, "result": str(node.result)[:500]},
            "pending_nodes": [{"id": n.id, "goal": n.goal} for n in self.nodes.values() if n.status == NodeStatus.PENDING],
            "completed_nodes": [{"id": n.id, "goal": n.goal, "status": n.status.value} for n in self.nodes.values() if n.status in (NodeStatus.COMPLETED, NodeStatus.FAILED)],
        }

        try:
            response = await adapter.generate(
                system_prompt=(
                    "You are an agent reflection engine. After a node executed, decide "
                    "if the execution graph should be modified. Options:\n"
                    "1. NO_CHANGE — continue as planned\n"
                    "2. SPAWN — add a new sub-task (provide: goal, tool_name, tool_params)\n"
                    "3. KILL — remove a pending node that's no longer needed (provide: target_node_id)\n"
                    "4. RETRY_ALT — the node failed, try an alternative approach (provide: new goal/tool)\n\n"
                    "Respond with JSON: {\"action\": \"NO_CHANGE|SPAWN|KILL|RETRY_ALT\", ...details}\n"
                    "Only suggest mutations when genuinely needed. Most of the time, NO_CHANGE is correct."
                ),
                user_prompt=json.dumps(graph_state, default=str),
                temperature=0.2,
                max_tokens=300,
            )
            
            content = response.get("content", "")
            decision = self._parse_json_object(content)
            action = decision.get("action", "NO_CHANGE")
            
            if action == "SPAWN" and decision.get("goal"):
                new_node = GraphNode(
                    goal=decision["goal"],
                    tool_name=decision.get("tool_name", ""),
                    tool_params=decision.get("tool_params", {}),
                    node_type=decision.get("type", "action"),
                )
                mutation = GraphMutation(
                    type=MutationType.SPAWN,
                    reason=decision.get("reason", "discovered new sub-task"),
                    new_node=new_node,
                    insert_after=node.id,
                )
                await self._apply_mutation(mutation, agent_id)

            elif action == "KILL" and decision.get("target_node_id"):
                mutation = GraphMutation(
                    type=MutationType.KILL,
                    reason=decision.get("reason", "no longer needed"),
                    target_node_id=decision["target_node_id"],
                )
                await self._apply_mutation(mutation, agent_id)

            elif action == "RETRY_ALT" and node.status == NodeStatus.FAILED:
                new_node = GraphNode(
                    goal=decision.get("goal", node.goal),
                    tool_name=decision.get("tool_name", ""),
                    tool_params=decision.get("tool_params", {}),
                    node_type="action",
                )
                mutation = GraphMutation(
                    type=MutationType.RETRY_ALT,
                    reason=decision.get("reason", "trying alternative approach"),
                    target_node_id=node.id,
                    new_node=new_node,
                )
                await self._apply_mutation(mutation, agent_id)

        except Exception as e:
            logger.debug(f"Reflection failed (non-critical): {e}")

    async def _apply_mutation(self, mutation: GraphMutation, agent_id: str):
        """Apply a graph mutation (with security check)."""
        # Log the mutation
        if self.audit:
            self.audit.log({
                "type": "graph_mutation",
                "mutation_type": mutation.type.value,
                "reason": mutation.reason,
                "agent_id": agent_id,
            })

        try:
            if mutation.type == MutationType.SPAWN and mutation.new_node:
                self.add_node(mutation.new_node, after=mutation.insert_after)
                self.mutations_applied += 1
                logger.info(f"SPAWN: Added node '{mutation.new_node.goal[:40]}' after {mutation.insert_after}")

            elif mutation.type == MutationType.KILL and mutation.target_node_id:
                removed = self.remove_branch(mutation.target_node_id)
                self.mutations_applied += 1
                logger.info(f"KILL: Removed {len(removed)} nodes starting from {mutation.target_node_id}")

            elif mutation.type == MutationType.RETRY_ALT and mutation.new_node and mutation.target_node_id:
                self.replace_node(mutation.target_node_id, mutation.new_node)
                self.mutations_applied += 1
                logger.info(f"RETRY_ALT: Replaced {mutation.target_node_id} with '{mutation.new_node.goal[:40]}'")

        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            self.mutations_blocked += 1

    # ── Helper Methods ────────────────────────────────────────────

    def _get_executable_nodes(self) -> list[GraphNode]:
        """Get nodes whose parents are all completed (ready to run)."""
        executable = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            if not node.parent_ids:
                executable.append(node)
            elif all(
                self.nodes.get(pid) and self.nodes[pid].status in (NodeStatus.COMPLETED, NodeStatus.KILLED)
                for pid in node.parent_ids
            ):
                executable.append(node)
        return executable

    def has_pending_nodes(self) -> bool:
        return any(n.status == NodeStatus.PENDING for n in self.nodes.values())

    async def _reason_node(self, node: GraphNode, agent_id: str) -> str:
        """Use LLM to reason about a node (no tool call)."""
        adapter = None
        if self.model_registry:
            adapter = self.model_registry.get_adapter("reasoning")
        if not adapter:
            return f"[No LLM available] Goal: {node.goal}"

        context_data = ""
        # Gather parent results as context
        for pid in node.parent_ids:
            if pid in self.nodes and self.nodes[pid].result:
                context_data += f"\nInput from previous step: {str(self.nodes[pid].result)[:500]}"

        response = await adapter.generate(
            system_prompt="You are an intelligent agent executing a sub-task. Provide a concise, actionable result.",
            user_prompt=f"Task: {node.goal}{context_data}",
            temperature=0.4,
            max_tokens=500,
        )
        return response.get("content", "")

    async def _synthesize_node(self, node: GraphNode, parent_results: dict, agent_id: str) -> str:
        """Synthesize results from multiple parent nodes."""
        adapter = None
        if self.model_registry:
            adapter = self.model_registry.get_adapter("reasoning")
        if not adapter:
            return json.dumps(parent_results, default=str)

        response = await adapter.generate(
            system_prompt="Synthesize the following results into a coherent summary.",
            user_prompt=f"Goal: {node.goal}\n\nInputs:\n{json.dumps(parent_results, default=str, indent=2)}",
            temperature=0.3,
            max_tokens=800,
        )
        return response.get("content", "")

    async def _synthesize_results(self, goal: str, agent_id: str) -> Any:
        """Synthesize all leaf node results into a final output."""
        leaf_results = {}
        for node in self.nodes.values():
            if node.status == NodeStatus.COMPLETED and not node.child_ids:
                leaf_results[node.id] = {"goal": node.goal, "result": node.result}

        if not leaf_results:
            return None

        adapter = None
        if self.model_registry:
            adapter = self.model_registry.get_adapter("reasoning")
        if not adapter:
            return leaf_results

        try:
            response = await adapter.generate(
                system_prompt="Synthesize all sub-task results into a final comprehensive answer.",
                user_prompt=f"Original goal: {goal}\n\nResults:\n{json.dumps(leaf_results, default=str, indent=2)}",
                temperature=0.3,
                max_tokens=1000,
            )
            return response.get("content", "")
        except Exception:
            return leaf_results

    def _parse_json_array(self, text: str) -> list[dict]:
        """Extract a JSON array from LLM output."""
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return [{"goal": text, "type": "action"}]

    def _parse_json_object(self, text: str) -> dict:
        """Extract a JSON object from LLM output."""
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"action": "NO_CHANGE"}

    def _node_to_trace(self, node: GraphNode) -> dict:
        return {
            "id": node.id,
            "goal": node.goal,
            "type": node.node_type,
            "status": node.status.value,
            "tool": node.tool_name,
            "duration_ms": node.duration_ms,
            "result_preview": str(node.result)[:200] if node.result else None,
            "error": node.error,
            "parents": node.parent_ids,
            "children": node.child_ids,
            "attempt": node.attempt,
        }

    def get_graph_state(self) -> dict:
        """Get the current graph state for dashboard visualization."""
        return {
            "nodes": [self._node_to_trace(n) for n in self.nodes.values()],
            "root_ids": self.root_ids,
            "mutations_applied": self.mutations_applied,
            "mutations_blocked": self.mutations_blocked,
            "total_nodes": len(self.nodes),
            "pending": sum(1 for n in self.nodes.values() if n.status == NodeStatus.PENDING),
            "running": sum(1 for n in self.nodes.values() if n.status == NodeStatus.RUNNING),
            "completed": sum(1 for n in self.nodes.values() if n.status == NodeStatus.COMPLETED),
            "failed": sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED),
        }

    # ── Events ────────────────────────────────────────────────────

    def add_listener(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def _emit(self, event_type: str, data: dict):
        event = {"type": f"cognitive_graph.{event_type}", "data": data, "timestamp": time.time()}
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass
