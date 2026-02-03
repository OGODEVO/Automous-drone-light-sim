"""Fake perception stub with noise knobs."""

from dataclasses import dataclass
from collections import deque
from typing import Callable
import random

from sim.config import SimConfig
from sim.drone import DroneState
from sim.world import Course


@dataclass
class PerceptionResult:
    """What the brain 'sees' each tick."""

    # Gate info (always present if visible)
    gate_bearing: float  # Angle to gate relative to drone yaw (radians)
    gate_distance: float  # Distance to gate (meters)
    gate_elevation: float = 0.0  # Pitch angle to gate (radians, + = above)

    # Obstacle info (None if no obstacle nearby)
    obstacle_bearing: float | None = None  # Angle to nearest obstacle
    obstacle_distance: float | None = None  # Distance to nearest obstacle


def create_perception(config: SimConfig) -> Callable[[DroneState, Course, SimConfig], PerceptionResult | None]:
    """
    Create a perception function with the configured noise.

    Returns a closure that:
    - Returns None if no gate or dropout occurred (brain must handle blindness)
    - Returns PerceptionResult with noisy bearing/distance if visible
    - Includes obstacle info if an obstacle is within detection range
    """
    # Delay buffer for perception lag
    delay_buffer: deque[PerceptionResult | None] = deque(maxlen=max(1, config.noise_delay_ticks + 1))
    obstacle_detection_range = 15.0  # meters - need early warning at high speed

    def perceive(state: DroneState, course: Course, cfg: SimConfig) -> PerceptionResult | None:
        gate = course.get_current_gate()

        if gate is None:
            # No more gates - return None (course complete)
            result = None
        elif random.random() < cfg.noise_dropout:
            # Dropout - return None (blind this tick)
            result = None
        else:
            # Visible - compute noisy measurements for gate
            true_bearing = gate.bearing_from(state)
            true_distance = gate.distance_to(state)
            true_elevation = gate.elevation_from(state)

            noisy_bearing = true_bearing + random.gauss(0, cfg.noise_jitter)
            noisy_distance = max(0.1, true_distance + random.gauss(0, cfg.noise_jitter * 2))
            noisy_elevation = true_elevation + random.gauss(0, cfg.noise_jitter)

            # Check for nearby obstacles
            obstacle_bearing = None
            obstacle_distance = None
            nearest_obs, obs_dist = course.get_nearest_obstacle(state)
            if nearest_obs and obs_dist < obstacle_detection_range:
                obstacle_bearing = nearest_obs.bearing_from(state) + random.gauss(0, cfg.noise_jitter)
                obstacle_distance = max(0.1, obs_dist + random.gauss(0, cfg.noise_jitter * 2))

            result = PerceptionResult(
                gate_bearing=noisy_bearing,
                gate_distance=noisy_distance,
                gate_elevation=noisy_elevation,
                obstacle_bearing=obstacle_bearing,
                obstacle_distance=obstacle_distance,
            )

        # Apply delay
        delay_buffer.append(result)

        if cfg.noise_delay_ticks > 0 and len(delay_buffer) > cfg.noise_delay_ticks:
            return delay_buffer[0]
        elif cfg.noise_delay_ticks == 0:
            return result
        else:
            # Not enough history yet - blind
            return None

    return perceive
