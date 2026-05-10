from __future__ import annotations

from enum import Flag, auto


class Capability(Flag):
    PREDICTIONS = auto()
    EMBEDDINGS = auto()
    EFFORT = auto()


class CapabilityMissing(Exception):
    def __init__(self, method_name: str, cap: Capability):
        self.method_name = method_name
        self.cap = cap
        super().__init__(f"{method_name} does not support {cap.name}")
