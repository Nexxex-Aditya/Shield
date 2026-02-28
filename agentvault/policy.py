"""
AgentVault — Policy Engine
YAML-based policy parser & evaluator with default-deny, wildcards, rate limits,
time restrictions, and hot-reload.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import yaml

from .models import (
    AgentAction,
    Decision,
    FirewallDecision,
    LogLevel,
    PolicyConfig,
    PolicyRule,
    RateLimitConfig,
)

logger = logging.getLogger("agentvault.policy")


class RateLimiter:
    """Sliding-window rate limiter per agent per action."""

    def __init__(self) -> None:
        # key: (agent_id, action) -> list of timestamps
        self._windows: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, agent_id: str, action: str, config: RateLimitConfig) -> bool:
        """Return True if the action is within rate limit, False if exceeded."""
        key = (agent_id, action)
        now = time.time()
        cutoff = now - config.window

        with self._lock:
            # Prune expired entries
            self._windows[key] = [t for t in self._windows[key] if t > cutoff]

            if len(self._windows[key]) >= config.max:
                return False

            self._windows[key].append(now)
            return True

    def get_count(self, agent_id: str, action: str, window: int) -> int:
        """Get current count in window."""
        key = (agent_id, action)
        now = time.time()
        cutoff = now - window
        with self._lock:
            return len([t for t in self._windows.get(key, []) if t > cutoff])

    def reset(self) -> None:
        """Reset all rate limit counters."""
        with self._lock:
            self._windows.clear()


class PolicyEngine:
    """
    YAML-based policy engine.
    
    - Loads policies from YAML files
    - Matches actions using exact name, wildcards (fnmatch), or regex
    - Enforces rate limits via sliding window
    - Supports time-of-day and day-of-week restrictions
    - Default-deny: any unmatched action is DENIED
    - Hot-reload: watches policy file for changes
    """

    def __init__(self) -> None:
        self._policies: list[PolicyConfig] = []
        self._rate_limiter = RateLimiter()
        self._policy_path: Optional[str] = None
        self._last_modified: float = 0
        self._lock = threading.Lock()
        self._rule_hit_counts: dict[str, int] = defaultdict(int)

    def load(self, path: str) -> None:
        """Load policies from a YAML file."""
        self._policy_path = os.path.abspath(path)
        self._reload()

    def _reload(self) -> None:
        """Internal reload from the stored path."""
        if not self._policy_path or not os.path.exists(self._policy_path):
            logger.warning("Policy file not found: %s", self._policy_path)
            return

        try:
            mtime = os.path.getmtime(self._policy_path)
            with open(self._policy_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            if raw is None:
                logger.warning("Empty policy file: %s", self._policy_path)
                return

            # Parse into PolicyConfig
            policies = []
            if isinstance(raw, dict):
                raw = [raw]

            for entry in raw:
                rules = []
                for rule_data in entry.get("rules", []):
                    # Map decision string to enum
                    decision_str = rule_data.get("decision", "deny").upper()
                    decision = Decision(decision_str)

                    rate_limit = None
                    if "rate_limit" in rule_data:
                        rate_limit = RateLimitConfig(**rule_data["rate_limit"])

                    sandbox_cfg = None
                    if "sandbox" in rule_data:
                        from .models import SandboxRuleConfig
                        sandbox_cfg = SandboxRuleConfig(**rule_data["sandbox"])

                    log_level_str = rule_data.get("log_level", "standard")
                    log_level = LogLevel(log_level_str)

                    rules.append(PolicyRule(
                        action=rule_data.get("action", "*"),
                        decision=decision,
                        rate_limit=rate_limit,
                        sandbox=sandbox_cfg,
                        log_level=log_level,
                        allowed_hours=rule_data.get("allowed_hours"),
                        allowed_days=rule_data.get("allowed_days"),
                        conditions=rule_data.get("conditions", {}),
                        description=rule_data.get("description", ""),
                    ))

                default_str = entry.get("default", "deny").upper()
                policies.append(PolicyConfig(
                    agent=entry.get("agent", "*"),
                    default=Decision(default_str),
                    rules=rules,
                    description=entry.get("description", ""),
                ))

            with self._lock:
                self._policies = policies
                self._last_modified = mtime

            logger.info(
                "Loaded %d policies with %d total rules from %s",
                len(policies),
                sum(len(p.rules) for p in policies),
                self._policy_path,
            )

        except Exception as e:
            logger.error("Failed to load policy file %s: %s", self._policy_path, e)
            raise

    def check_for_reload(self) -> bool:
        """Check if the policy file has been modified and reload if needed."""
        if not self._policy_path or not os.path.exists(self._policy_path):
            return False

        mtime = os.path.getmtime(self._policy_path)
        if mtime > self._last_modified:
            logger.info("Policy file changed, reloading...")
            self._reload()
            return True
        return False

    def evaluate(self, action: AgentAction) -> FirewallDecision:
        """
        Evaluate an agent action against loaded policies.
        Returns a FirewallDecision with ALLOW, DENY, or ESCALATE.
        """
        # Hot-reload check
        self.check_for_reload()

        with self._lock:
            policies = list(self._policies)

        if not policies:
            return FirewallDecision(
                decision=Decision.DENY,
                reasoning="No policies loaded — default deny",
                trace_id=action.trace_id,
            )

        # Find matching policy for this agent
        matching_policy = self._find_matching_policy(action.agent_id, policies)
        if not matching_policy:
            return FirewallDecision(
                decision=Decision.DENY,
                reasoning=f"No policy found for agent '{action.agent_id}' — default deny",
                trace_id=action.trace_id,
            )

        # Find matching rule
        matched_rule = self._find_matching_rule(action.action_name, matching_policy.rules)

        if not matched_rule:
            return FirewallDecision(
                decision=matching_policy.default,
                reasoning=f"No rule matched action '{action.action_name}' — default {matching_policy.default.value}",
                trace_id=action.trace_id,
            )

        # Track rule hits
        rule_key = f"{matching_policy.agent}:{matched_rule.action}"
        self._rule_hit_counts[rule_key] += 1

        # Check time restrictions
        time_check = self._check_time_restrictions(matched_rule)
        if time_check is not None:
            return FirewallDecision(
                decision=Decision.DENY,
                reasoning=time_check,
                trace_id=action.trace_id,
                matched_rule=matched_rule.action,
            )

        # Check rate limits
        if matched_rule.rate_limit:
            if not self._rate_limiter.check(
                action.agent_id, action.action_name, matched_rule.rate_limit
            ):
                return FirewallDecision(
                    decision=Decision.DENY,
                    reasoning=(
                        f"Rate limit exceeded for '{action.action_name}': "
                        f"max {matched_rule.rate_limit.max} per {matched_rule.rate_limit.window}s"
                    ),
                    trace_id=action.trace_id,
                    matched_rule=matched_rule.action,
                )

        # Check parameter conditions
        condition_check = self._check_conditions(matched_rule, action)
        if condition_check is not None:
            return condition_check

        # Return the rule's decision
        return FirewallDecision(
            decision=matched_rule.decision,
            reasoning=f"Matched rule '{matched_rule.action}' → {matched_rule.decision.value}",
            trace_id=action.trace_id,
            matched_rule=matched_rule.action,
        )

    def _find_matching_policy(
        self, agent_id: str, policies: list[PolicyConfig]
    ) -> Optional[PolicyConfig]:
        """Find the best matching policy for an agent."""
        # First try exact match
        for policy in policies:
            if policy.agent == agent_id:
                return policy

        # Then try wildcard match
        for policy in policies:
            if policy.agent == "*":
                return policy
            if fnmatch.fnmatch(agent_id, policy.agent):
                return policy

        return None

    def _find_matching_rule(
        self, action_name: str, rules: list[PolicyRule]
    ) -> Optional[PolicyRule]:
        """Find the best matching rule. Exact > wildcard > regex > semantic."""
        # Exact match first
        for rule in rules:
            if rule.action == action_name:
                return rule

        # Then wildcard (fnmatch)
        for rule in rules:
            if "*" in rule.action or "?" in rule.action:
                if fnmatch.fnmatch(action_name, rule.action):
                    return rule

        # Then regex (if rule starts with ^)
        for rule in rules:
            if rule.action.startswith("^"):
                try:
                    if re.match(rule.action, action_name):
                        return rule
                except re.error:
                    logger.warning("Invalid regex in rule: %s", rule.action)

        # Semantic matching — catch synonym evasion
        semantic_match = self._find_semantic_match(action_name, rules)
        if semantic_match:
            return semantic_match

        return None

    # Synonym groups: actions in the same group are treated as equivalent
    SEMANTIC_GROUPS = [
        {"delete", "remove", "erase", "purge", "destroy", "wipe", "drop", "truncate"},
        {"read", "get", "fetch", "retrieve", "load", "query", "select", "view"},
        {"write", "set", "put", "save", "store", "update", "insert", "upsert"},
        {"send", "post", "push", "transmit", "deliver", "dispatch", "emit", "publish"},
        {"create", "add", "new", "make", "generate", "init", "spawn"},
        {"execute", "run", "eval", "invoke", "call", "launch", "start"},
        {"list", "enumerate", "scan", "browse", "index", "catalog"},
        {"modify", "edit", "change", "alter", "patch", "mutate"},
        {"admin", "sudo", "root", "elevate", "privilege"},
    ]

    def _find_semantic_match(
        self, action_name: str, rules: list[PolicyRule]
    ) -> Optional[PolicyRule]:
        """Find a rule that semantically matches the action (synonym-aware)."""
        # Extract the verb from the action name (e.g., "remove_file" → "remove")
        action_verb = action_name.split("_")[0].lower()
        action_suffix = "_".join(action_name.split("_")[1:]) if "_" in action_name else ""

        # Find which synonym group the verb belongs to
        verb_group = None
        for group in self.SEMANTIC_GROUPS:
            if action_verb in group:
                verb_group = group
                break

        if not verb_group:
            return None

        # Check if any rule's action verb is in the same synonym group
        for rule in rules:
            rule_verb = rule.action.replace("*", "").replace("?", "").split("_")[0].lower()
            if rule_verb in verb_group and rule_verb != action_verb:
                # Semantic match found — the action is trying to bypass the rule
                logger.info(
                    "Semantic match: '%s' matched rule '%s' (verb '%s' ≈ '%s')",
                    action_name, rule.action, action_verb, rule_verb,
                )
                return rule

        return None


    def _check_time_restrictions(self, rule: PolicyRule) -> Optional[str]:
        """Check if current time falls within allowed hours/days. Returns denial reason or None."""
        now = datetime.now()

        if rule.allowed_hours:
            try:
                parts = rule.allowed_hours.split("-")
                start_hour = int(parts[0])
                end_hour = int(parts[1])
                if not (start_hour <= now.hour < end_hour):
                    return (
                        f"Action '{rule.action}' denied: current hour {now.hour} "
                        f"is outside allowed hours {rule.allowed_hours}"
                    )
            except (ValueError, IndexError):
                logger.warning("Invalid allowed_hours format: %s", rule.allowed_hours)

        if rule.allowed_days:
            day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            current_day = day_names[now.weekday()]
            try:
                parts = rule.allowed_days.lower().split("-")
                start_idx = day_names.index(parts[0])
                end_idx = day_names.index(parts[1])
                if start_idx <= end_idx:
                    allowed = start_idx <= now.weekday() <= end_idx
                else:
                    allowed = now.weekday() >= start_idx or now.weekday() <= end_idx
                if not allowed:
                    return (
                        f"Action '{rule.action}' denied: current day '{current_day}' "
                        f"is outside allowed days {rule.allowed_days}"
                    )
            except (ValueError, IndexError):
                logger.warning("Invalid allowed_days format: %s", rule.allowed_days)

        return None

    def _check_conditions(
        self, rule: PolicyRule, action: AgentAction
    ) -> Optional[FirewallDecision]:
        """Check parameter-level conditions. Returns a DENY decision if violated, None otherwise."""
        conditions = rule.conditions

        if not conditions:
            return None

        for key, value in conditions.items():
            if key == "if_param_exceeds":
                # value = {"param_name": threshold}
                if isinstance(value, dict):
                    for param_name, threshold in value.items():
                        param_val = action.parameters.get(param_name)
                        if param_val is not None and float(param_val) > float(threshold):
                            return FirewallDecision(
                                decision=Decision.ESCALATE,
                                reasoning=(
                                    f"Parameter '{param_name}' value {param_val} "
                                    f"exceeds threshold {threshold}"
                                ),
                                trace_id=action.trace_id,
                                matched_rule=rule.action,
                            )

            elif key == "if_param_contains":
                # value = {"param_name": "substring"}
                if isinstance(value, dict):
                    for param_name, substring in value.items():
                        param_val = str(action.parameters.get(param_name, ""))
                        if substring.lower() in param_val.lower():
                            return FirewallDecision(
                                decision=Decision.DENY,
                                reasoning=(
                                    f"Parameter '{param_name}' contains "
                                    f"restricted content '{substring}'"
                                ),
                                trace_id=action.trace_id,
                                matched_rule=rule.action,
                            )

        return None

    @property
    def policies(self) -> list[PolicyConfig]:
        """Get loaded policies."""
        with self._lock:
            return list(self._policies)

    @property
    def rule_hit_counts(self) -> dict[str, int]:
        """Get hit counts per rule."""
        return dict(self._rule_hit_counts)

    def get_rules_for_agent(self, agent_id: str) -> list[PolicyRule]:
        """Get all rules applicable to an agent."""
        with self._lock:
            policy = self._find_matching_policy(agent_id, self._policies)
            if policy:
                return list(policy.rules)
            return []
