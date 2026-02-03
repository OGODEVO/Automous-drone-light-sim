"""Simulation configuration with noise knobs."""

from dataclasses import dataclass


@dataclass
class SimConfig:
    """Configuration for the simulation loop and noise parameters."""

    # Timing
    hz: int = 30  # Tick rate (Hz)

    # Physics limits (tuned for high-speed racing)
    max_speed: float = 30.0  # m/s (26 m/s target + headroom)
    max_accel: float = 15.0  # m/s² (aggressive acceleration)
    max_yaw_rate: float = 4.0  # rad/s (~230°/s, agile turns)

    # World bounds (larger for high speed)
    bounds_min: tuple[float, float, float] = (-100.0, -100.0, 0.0)
    bounds_max: tuple[float, float, float] = (300.0, 300.0, 50.0)

    # Noise knobs
    noise_dropout: float = 0.0  # Probability perception returns None (0-1)
    noise_jitter: float = 0.0  # Std dev for measurement noise
    noise_delay_ticks: int = 0  # Perception lag in ticks

    # Termination
    max_time: float = 60.0  # Max simulation time (seconds)

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        return 1.0 / self.hz
