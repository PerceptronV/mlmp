"""Bounded priority queue buffer for top-K programs."""

import heapq
import random

from ..ast_nodes import ASTNode
from ..enumeration.fingerprint import Fingerprint


class PriorityQueueBuffer:
    """
    Bounded buffer of the top-K highest-reward programs.

    Programs are stored as (reward, id, program_ast, trajectory, fingerprint) tuples.
    When the buffer is full, inserting a program with reward higher
    than the current minimum evicts the minimum.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: list[tuple[float, int, ASTNode, list, Fingerprint]] = []
        self.fingerprints: set[Fingerprint] = set()

    def insert(
        self,
        reward: float,
        program: ASTNode,
        trajectory: list,
        fingerprint: Fingerprint,
    ) -> bool:
        """Insert a program if it improves the buffer. Returns True if inserted."""
        if fingerprint in self.fingerprints:
            return False

        entry = (reward, id(program), program, trajectory, fingerprint)

        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, entry)
            self.fingerprints.add(fingerprint)
            return True

        if reward > self.buffer[0][0]:
            evicted = heapq.heapreplace(self.buffer, entry)
            self.fingerprints.discard(evicted[4])
            self.fingerprints.add(fingerprint)
            return True

        return False

    def sample(self, batch_size: int) -> list[tuple[float, ASTNode, list]]:
        """Sample a batch of (reward, program, trajectory) tuples."""
        indices = random.sample(
            range(len(self.buffer)), min(batch_size, len(self.buffer)),
        )
        return [
            (self.buffer[i][0], self.buffer[i][2], self.buffer[i][3])
            for i in indices
        ]

    def min_reward(self) -> float:
        if not self.buffer:
            return 0.0
        return self.buffer[0][0]

    def __len__(self):
        return len(self.buffer)
