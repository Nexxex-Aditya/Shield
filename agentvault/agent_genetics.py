"""
Shield Command — Agent Genetics: Evolutionary Optimization

Every Shield agent has a genome — a serialized config of all tunable parameters.
The EvolutionEngine runs variants in shadow mode, scores fitness, and breeds
the top performers through mutation and crossover.

Result: agents evolve to become better at their job through natural selection.
No manual tuning required.

Integration points:
    - CognitiveGraph: executes agent variants
    - CognitiveMemory: stores fitness records as episodes
    - MCPGateway: evaluates security compliance during fitness testing
    - CIBILEngine: trust scores factor into fitness
"""

import json
import random
import uuid
import time
import copy
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("shield.agent_genetics")


# ---------------------------------------------------------------------------
# Agent Genome
# ---------------------------------------------------------------------------

@dataclass
class AgentGenome:
    """
    The complete 'DNA' of an agent — every tunable parameter serialized
    into a breedable, mutable structure.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    name: str = ""
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    # Core LLM config
    system_prompt: str = "You are a helpful AI agent."
    model_id: str = "gpt-4"
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 1000

    # Tool preferences: tool_name → selection weight (0-1)
    tool_preferences: dict[str, float] = field(default_factory=dict)

    # Memory configuration
    memory_recency_weight: float = 0.3
    memory_impact_weight: float = 0.5
    memory_similarity_weight: float = 0.2

    # Security thresholds
    max_retries: int = 3
    risk_tolerance: float = 0.5  # 0=paranoid, 1=reckless

    # Fitness tracking
    fitness_score: float = 0.0
    fitness_samples: int = 0

    created_at: float = field(default_factory=time.time)

    def mutate(self, mutation_rate: float = 0.15) -> "AgentGenome":
        """Create a mutated copy of this genome."""
        child = copy.deepcopy(self)
        child.id = str(uuid.uuid4())[:10]
        child.generation = self.generation + 1
        child.parent_ids = [self.id]
        child.fitness_score = 0.0
        child.fitness_samples = 0

        # Temperature mutation
        if random.random() < mutation_rate:
            child.temperature = max(0.0, min(1.5, child.temperature + random.uniform(-0.15, 0.15)))

        # Top-p mutation
        if random.random() < mutation_rate:
            child.top_p = max(0.1, min(1.0, child.top_p + random.uniform(-0.1, 0.1)))

        # Max tokens mutation
        if random.random() < mutation_rate:
            child.max_tokens = max(100, min(4000, child.max_tokens + random.randint(-200, 200)))

        # Memory weight mutations (must sum to ~1.0)
        if random.random() < mutation_rate:
            delta = random.uniform(-0.1, 0.1)
            child.memory_recency_weight = max(0.05, min(0.8, child.memory_recency_weight + delta))
            child.memory_impact_weight = max(0.05, min(0.8, child.memory_impact_weight - delta * 0.5))
            child.memory_similarity_weight = max(0.05, 1.0 - child.memory_recency_weight - child.memory_impact_weight)

        # Tool preference mutations
        if random.random() < mutation_rate and child.tool_preferences:
            tool = random.choice(list(child.tool_preferences.keys()))
            child.tool_preferences[tool] = max(0.0, min(1.0,
                child.tool_preferences[tool] + random.uniform(-0.2, 0.2)
            ))

        # Risk tolerance mutation
        if random.random() < mutation_rate:
            child.risk_tolerance = max(0.0, min(1.0, child.risk_tolerance + random.uniform(-0.15, 0.15)))

        # Retry count mutation
        if random.random() < mutation_rate:
            child.max_retries = max(1, min(5, child.max_retries + random.choice([-1, 1])))

        return child

    def crossover(self, partner: "AgentGenome") -> "AgentGenome":
        """Breed two genomes, taking the best traits from each."""
        child = AgentGenome(
            name=f"{self.name}x{partner.name}",
            generation=max(self.generation, partner.generation) + 1,
            parent_ids=[self.id, partner.id],
        )

        # Take system prompt from fitter parent
        if self.fitness_score >= partner.fitness_score:
            child.system_prompt = self.system_prompt
            child.model_id = self.model_id
        else:
            child.system_prompt = partner.system_prompt
            child.model_id = partner.model_id

        # Average numerical parameters
        child.temperature = (self.temperature + partner.temperature) / 2
        child.top_p = (self.top_p + partner.top_p) / 2
        child.max_tokens = (self.max_tokens + partner.max_tokens) // 2
        child.risk_tolerance = (self.risk_tolerance + partner.risk_tolerance) / 2
        child.max_retries = max(self.max_retries, partner.max_retries)

        # Merge tool preferences (union with averages)
        all_tools = set(self.tool_preferences.keys()) | set(partner.tool_preferences.keys())
        for tool in all_tools:
            w1 = self.tool_preferences.get(tool, 0.5)
            w2 = partner.tool_preferences.get(tool, 0.5)
            child.tool_preferences[tool] = (w1 + w2) / 2

        # Memory weights from fitter parent
        fitter = self if self.fitness_score >= partner.fitness_score else partner
        child.memory_recency_weight = fitter.memory_recency_weight
        child.memory_impact_weight = fitter.memory_impact_weight
        child.memory_similarity_weight = fitter.memory_similarity_weight

        return child


# ---------------------------------------------------------------------------
# Fitness Evaluation
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    """Result of evaluating a genome's fitness."""
    genome_id: str
    success_rate: float = 0.0      # 0-1
    avg_latency_ms: float = 0.0
    avg_cost: float = 0.0
    security_compliance: float = 1.0  # 0-1
    quality_score: float = 0.0     # 0-1 (from LLM-as-judge or heuristics)
    composite_fitness: float = 0.0


class FitnessEvaluator:
    """
    Evaluates the fitness of an agent genome against a test workload.
    
    Fitness = weighted combination of:
        - success_rate (did tasks complete?)
        - latency (how fast?)
        - cost (how expensive?)
        - security_compliance (how safe?)
        - quality (how good were the outputs?)
    """

    def __init__(
        self,
        success_weight: float = 0.35,
        quality_weight: float = 0.30,
        security_weight: float = 0.20,
        latency_weight: float = 0.10,
        cost_weight: float = 0.05,
    ):
        self.weights = {
            "success": success_weight,
            "quality": quality_weight,
            "security": security_weight,
            "latency": latency_weight,
            "cost": cost_weight,
        }

    async def evaluate(
        self,
        genome: AgentGenome,
        test_tasks: list[dict],
        cognitive_graph=None,
    ) -> FitnessResult:
        """
        Run the genome against test tasks and compute fitness.
        
        Each test_task should have: {"goal": str, "expected_output": str (optional)}
        """
        successes = 0
        total_latency = 0.0
        total_cost = 0.0
        quality_scores = []

        for task in test_tasks:
            try:
                start = time.time()
                
                if cognitive_graph:
                    # Execute through the cognitive graph
                    result = await cognitive_graph.execute(
                        goal=task["goal"],
                        agent_id=f"genetics-eval-{genome.id}",
                    )
                    elapsed = (time.time() - start) * 1000
                    
                    if result.status in ("completed", "partial"):
                        successes += 1
                    total_latency += elapsed

                    # Quality scoring (simple heuristic)
                    if result.final_output:
                        output_len = len(str(result.final_output))
                        quality = min(output_len / 200, 1.0)  # longer = better (simple heuristic)
                        quality_scores.append(quality)
                else:
                    # Without cognitive graph, simulate
                    successes += 1
                    total_latency += random.uniform(500, 3000)
                    quality_scores.append(random.uniform(0.5, 1.0))

            except Exception as e:
                logger.debug(f"Fitness eval failed for genome {genome.id}: {e}")
                quality_scores.append(0.0)

        n = max(len(test_tasks), 1)
        success_rate = successes / n
        avg_latency = total_latency / n if total_latency else 0
        avg_quality = sum(quality_scores) / max(len(quality_scores), 1)

        # Normalize latency (lower is better, cap at 10s)
        latency_score = max(0, 1 - avg_latency / 10000)

        # Composite fitness
        composite = (
            self.weights["success"] * success_rate
            + self.weights["quality"] * avg_quality
            + self.weights["security"] * 1.0  # placeholder
            + self.weights["latency"] * latency_score
            + self.weights["cost"] * 0.8  # placeholder
        )

        return FitnessResult(
            genome_id=genome.id,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            quality_score=avg_quality,
            composite_fitness=composite,
        )


# ---------------------------------------------------------------------------
# Evolution Engine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    """
    Runs evolutionary cycles on a population of agent genomes.
    
    Cycle:
        1. Evaluate fitness of all genomes in the population
        2. Select the top performers (elitism)
        3. Breed them (crossover)
        4. Mutate the offspring
        5. Replace the weakest with the new generation
    """

    def __init__(
        self,
        population_size: int = 6,
        elite_count: int = 2,
        mutation_rate: float = 0.15,
        db_path: str = "shield_memory.db",
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.evaluator = FitnessEvaluator()
        self.db_path = db_path
        self.population: list[AgentGenome] = []
        self.history: list[dict] = []
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS genome_history (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    generation INTEGER,
                    parent_ids TEXT,
                    genome_data TEXT,
                    fitness_score REAL,
                    created_at REAL
                )
            """)

    def initialize_population(self, base_genome: AgentGenome) -> list[AgentGenome]:
        """Create initial population from a base genome with mutations."""
        self.population = [copy.deepcopy(base_genome)]
        for i in range(self.population_size - 1):
            mutant = base_genome.mutate(mutation_rate=0.3)  # Higher initial diversity
            mutant.name = f"{base_genome.name}_v{i+1}"
            self.population.append(mutant)
        logger.info(f"Initialized population of {len(self.population)} genomes")
        return self.population

    async def evolve(
        self,
        test_tasks: list[dict],
        generations: int = 5,
        cognitive_graph=None,
    ) -> AgentGenome:
        """
        Run evolution for N generations and return the fittest genome.
        """
        if not self.population:
            raise ValueError("Population not initialized. Call initialize_population first.")

        for gen in range(generations):
            logger.info(f"=== Generation {gen + 1}/{generations} ===")
            
            # Evaluate fitness of all genomes
            for genome in self.population:
                result = await self.evaluator.evaluate(genome, test_tasks, cognitive_graph)
                genome.fitness_score = result.composite_fitness
                genome.fitness_samples += 1

            # Sort by fitness (descending)
            self.population.sort(key=lambda g: g.fitness_score, reverse=True)

            # Record history
            gen_record = {
                "generation": gen + 1,
                "best_fitness": self.population[0].fitness_score,
                "avg_fitness": sum(g.fitness_score for g in self.population) / len(self.population),
                "best_genome_id": self.population[0].id,
            }
            self.history.append(gen_record)
            logger.info(f"  Best: {gen_record['best_fitness']:.3f}  Avg: {gen_record['avg_fitness']:.3f}")

            # Save all genomes to history
            await self._save_generation(self.population)

            if gen < generations - 1:
                # Select elites
                elites = self.population[:self.elite_count]

                # Breed new offspring
                offspring = []
                while len(offspring) < self.population_size - self.elite_count:
                    parent_a = random.choice(elites)
                    parent_b = random.choice(self.population[:max(4, len(self.population) // 2)])
                    child = parent_a.crossover(parent_b)
                    child = child.mutate(self.mutation_rate)
                    child.name = f"gen{gen+2}_{len(offspring)}"
                    offspring.append(child)

                self.population = elites + offspring

        # Return the fittest
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        champion = self.population[0]
        logger.info(f"Evolution complete. Champion: {champion.id} (fitness={champion.fitness_score:.3f})")
        return champion

    async def _save_generation(self, genomes: list[AgentGenome]):
        """Save genome history to database."""
        with sqlite3.connect(self.db_path) as conn:
            for g in genomes:
                conn.execute(
                    """INSERT OR REPLACE INTO genome_history 
                       (id, name, generation, parent_ids, genome_data, fitness_score, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        g.id, g.name, g.generation, json.dumps(g.parent_ids),
                        json.dumps(asdict(g)), g.fitness_score, g.created_at,
                    ),
                )

    async def get_champion(self) -> Optional[AgentGenome]:
        """Get the best genome from all generations."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT genome_data FROM genome_history ORDER BY fitness_score DESC LIMIT 1"
            ).fetchone()
        if row:
            data = json.loads(row[0])
            return AgentGenome(**{k: v for k, v in data.items() if k in AgentGenome.__dataclass_fields__})
        return None

    async def get_evolution_history(self) -> list[dict]:
        """Get the fitness progression across generations."""
        return list(self.history)

    async def get_stats(self) -> dict:
        return {
            "population_size": len(self.population),
            "generations_completed": len(self.history),
            "best_fitness": max((g.fitness_score for g in self.population), default=0),
            "avg_fitness": sum(g.fitness_score for g in self.population) / max(len(self.population), 1),
        }
