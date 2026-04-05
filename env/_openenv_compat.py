"""
Compatibility shim replacing openenv.core base classes.

openenv 0.1.13 does not ship openenv.core — these lightweight replacements
provide the same interface used by models.py, environment.py, and client.py.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

# ── Action / Observation ──────────────────────────────────────────────────────

class Action(BaseModel):
    """Base class for environment actions."""

class Observation(BaseModel):
    """Base class for environment observations."""
    done: bool = False
    reward: float = 0.0
    metadata: dict = {}

# ── Environment ───────────────────────────────────────────────────────────────

A = TypeVar("A", bound=Action)
O = TypeVar("O", bound=Observation)
S = TypeVar("S")


class Environment(Generic[A, O, S]):
    """Minimal abstract base class for OpenEnv-style environments."""

    @abstractmethod
    def reset(self, **kwargs) -> O: ...

    @abstractmethod
    def step(self, action: A, **kwargs) -> O: ...

    @property
    @abstractmethod
    def state(self) -> S: ...

# ── EnvClient ─────────────────────────────────────────────────────────────────

class EnvClient(Generic[A, O, S]):
    """Stub base class — remote client not used during local training."""
