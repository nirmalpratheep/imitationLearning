"""
OpenEnv client for the car racing environment.

Async usage:
    async with RaceEnvClient(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(DriveAction(accel=1.0, steer=0.0))

Sync usage:
    with RaceEnvClient(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        result = client.step(DriveAction(accel=1.0, steer=0.0))
"""

from openenv.core import EnvClient
from .models import DriveAction, RaceObservation


class RaceEnvClient(EnvClient[DriveAction, RaceObservation, dict]):
    """Client for a running RaceEnvironment server."""
    pass
