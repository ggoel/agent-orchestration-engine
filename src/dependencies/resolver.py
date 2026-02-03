"""Dependency resolution for agent workflows."""

from collections import defaultdict, deque
from typing import Dict, List, Set

from ..agents.base import Agent


class DependencyResolver:
    """
    Resolves agent dependencies and creates execution plan.
    Uses topological sorting to determine execution order.
    """

    def __init__(self, agents: List[Agent]):
        """Initialize resolver with agents."""
        self.agents = {agent.name: agent for agent in agents}
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Validate that all dependencies exist and no cycles."""
        # Check all dependencies exist
        for agent in self.agents.values():
            for dep in agent.dependencies:
                if dep not in self.agents:
                    raise ValueError(
                        f"Agent {agent.name} depends on non-existent agent {dep}"
                    )

        # Check for cycles
        if self._has_cycle():
            raise ValueError("Circular dependency detected in workflow")

    def _has_cycle(self) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(agent_name: str) -> bool:
            visited.add(agent_name)
            rec_stack.add(agent_name)

            agent = self.agents[agent_name]
            for dep in agent.dependencies:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(agent_name)
            return False

        for agent_name in self.agents:
            if agent_name not in visited:
                if dfs(agent_name):
                    return True

        return False

    def get_execution_levels(self) -> List[List[Agent]]:
        """
        Get agents grouped by execution level using topological sort.

        Returns:
            List of agent groups where each group can execute in parallel
        """
        # Calculate in-degree for each agent
        in_degree: Dict[str, int] = defaultdict(int)
        for agent in self.agents.values():
            if agent.name not in in_degree:
                in_degree[agent.name] = 0
            for dep in agent.dependencies:
                in_degree[agent.name] += 1

        # Queue of agents with no dependencies
        queue = deque([
            self.agents[name]
            for name, degree in in_degree.items()
            if degree == 0
        ])

        levels: List[List[Agent]] = []

        while queue:
            # All agents in queue can execute in parallel
            level = list(queue)
            levels.append(level)

            # Process this level
            next_queue = deque()
            for agent in level:
                # Find agents that depend on this one
                for other_name, other_agent in self.agents.items():
                    if agent.name in other_agent.dependencies:
                        in_degree[other_name] -= 1
                        if in_degree[other_name] == 0:
                            next_queue.append(other_agent)

            queue = next_queue

        # Verify all agents were included
        total_agents = sum(len(level) for level in levels)
        if total_agents != len(self.agents):
            raise ValueError("Failed to resolve all dependencies")

        return levels

    def get_dependent_agents(self, agent_name: str) -> Set[str]:
        """Get all agents that depend on the given agent."""
        dependents = set()
        for name, agent in self.agents.items():
            if agent_name in agent.dependencies:
                dependents.add(name)
        return dependents

    def get_dependency_chain(self, agent_name: str) -> List[str]:
        """Get the full dependency chain for an agent (all ancestors)."""
        if agent_name not in self.agents:
            return []

        chain = []
        visited = set()

        def dfs(name: str):
            if name in visited:
                return
            visited.add(name)

            agent = self.agents[name]
            for dep in agent.dependencies:
                dfs(dep)
                if dep not in chain:
                    chain.append(dep)

        dfs(agent_name)
        return chain

    def get_critical_path(self) -> List[Agent]:
        """
        Get the critical path (longest chain of dependencies).

        Returns:
            List of agents in the critical path
        """
        def get_depth(agent_name: str, memo: Dict[str, int]) -> int:
            if agent_name in memo:
                return memo[agent_name]

            agent = self.agents[agent_name]
            if not agent.dependencies:
                memo[agent_name] = 1
                return 1

            max_depth = max(
                get_depth(dep, memo)
                for dep in agent.dependencies
            )
            memo[agent_name] = max_depth + 1
            return max_depth + 1

        memo: Dict[str, int] = {}
        depths = {name: get_depth(name, memo) for name in self.agents}

        # Find agent with maximum depth
        max_depth_agent = max(depths.items(), key=lambda x: x[1])[0]

        # Reconstruct path
        path = []
        current = max_depth_agent

        while current:
            path.append(self.agents[current])
            agent = self.agents[current]

            if not agent.dependencies:
                break

            # Find dependency with maximum depth
            current = max(
                agent.dependencies,
                key=lambda x: depths[x]
            )

        return list(reversed(path))
