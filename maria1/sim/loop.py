"""Fixed timestep simulation loop with logging."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Any
import json
import time

from .config import SimConfig
from .drone import DroneState, DroneCommand, update_dynamics
from .world import Course, is_out_of_bounds


@dataclass
class ScoreResult:
    """Final scoring after simulation completes."""

    time_elapsed: float = 0.0
    gates_passed: int = 0
    gates_total: int = 0
    gates_missed: int = 0
    collisions: int = 0
    out_of_bounds: bool = False
    completed: bool = False
    termination_reason: str = ""


class TickLogger:
    """Logs each tick for replay and debugging."""

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir
        self.entries: list[dict[str, Any]] = []
        self.config_snapshot: dict[str, Any] | None = None

    def set_config(self, config: SimConfig) -> None:
        """Store config snapshot for the log."""
        self.config_snapshot = asdict(config)

    def log_tick(
        self,
        tick: int,
        time: float,
        state: DroneState,
        obs: Any,
        fsm_state: str,
        cmd: DroneCommand,
    ) -> None:
        """Log a single tick."""
        self.entries.append(
            {
                "tick": tick,
                "t": round(time, 4),
                "state": {
                    "x": round(state.x, 3),
                    "y": round(state.y, 3),
                    "z": round(state.z, 3),
                    "yaw": round(state.yaw, 3),
                    "vx": round(state.vx, 3),
                    "vy": round(state.vy, 3),
                    "vz": round(state.vz, 3),
                },
                "obs": obs,
                "fsm": fsm_state,
                "cmd": {
                    "vx": round(cmd.vx, 3),
                    "vy": round(cmd.vy, 3),
                    "vz": round(cmd.vz, 3),
                    "yaw_rate": round(cmd.yaw_rate, 3),
                },
            }
        )

    def save(self, filename: str) -> Path | None:
        """Save log to JSON file."""
        if self.log_dir is None:
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.log_dir / filename

        log_data = {
            "config": self.config_snapshot,
            "tick_count": len(self.entries),
            "ticks": self.entries,
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        return filepath


def run_simulation(
    config: SimConfig,
    course: Course,
    brain: Callable[[Any], tuple[str, DroneCommand]],
    perception: Callable[[DroneState, Course, SimConfig], Any],
    initial_state: DroneState | None = None,
    logger: TickLogger | None = None,
    visualizer: Any = None,
    realtime: bool = False,
) -> tuple[ScoreResult, DroneState]:
    """
    Run the simulation loop.

    Args:
        config: Simulation configuration
        course: The course to fly
        brain: Function that takes observation and returns (fsm_state, command)
        perception: Function that takes (state, course, config) and returns observation
        initial_state: Starting drone state (defaults to origin)
        logger: Optional tick logger
        realtime: If True, sleep to maintain real-time speed

    Returns:
        Tuple of (ScoreResult, final DroneState)
    """
    state = initial_state if initial_state else DroneState()
    dt = config.dt
    tick = 0
    sim_time = 0.0

    result = ScoreResult(gates_total=course.gates_total())

    if logger:
        logger.set_config(config)

    wall_start = time.perf_counter()

    while True:
        # Check termination conditions
        if course.is_complete():
            result.completed = True
            result.termination_reason = "course_complete"
            break

        if sim_time >= config.max_time:
            result.termination_reason = "timeout"
            break

        if is_out_of_bounds(state, config):
            result.out_of_bounds = True
            result.termination_reason = "out_of_bounds"
            break

        # Perception
        obs = perception(state, course, config)

        # Brain decision
        fsm_state, cmd = brain(obs)

        # Log before state update
        if logger:
            obs_log = asdict(obs) if obs and hasattr(obs, "__dataclass_fields__") else obs
            logger.log_tick(tick, sim_time, state, obs_log, fsm_state, cmd)

        # Update visualizer
        if visualizer:
            visualizer.update(state)
            if hasattr(visualizer, "is_open") and not visualizer.is_open():
                result.termination_reason = "viz_closed"
                break

        # Update dynamics
        state = update_dynamics(state, cmd, config)

        # Check collision with obstacles
        if course.check_collision(state):
            result.collisions = course.collisions
            result.termination_reason = "collision"
            break

        # Check gate passage
        if course.check_gate_passed(state):
            course.advance_gate()

        # Advance time
        tick += 1
        sim_time += dt

        # Real-time pacing
        if realtime:
            target_wall = wall_start + sim_time
            now = time.perf_counter()
            if now < target_wall:
                time.sleep(target_wall - now)

    # Finalize result
    result.time_elapsed = sim_time
    result.gates_passed = course.gates_passed()
    result.gates_missed = course.gates_missed
    result.collisions = course.collisions

    return result, state
