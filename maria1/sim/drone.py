"""Drone state and dynamics."""

from dataclasses import dataclass
import math

from .config import SimConfig


@dataclass
class DroneState:
    """Minimal drone state for simulation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, 0 = +X axis

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0  # rad/s

    def copy(self) -> "DroneState":
        """Return a copy of this state."""
        return DroneState(
            x=self.x,
            y=self.y,
            z=self.z,
            yaw=self.yaw,
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
            yaw_rate=self.yaw_rate,
        )


@dataclass
class DroneCommand:
    """Command sent to the drone each tick."""

    vx: float = 0.0  # Desired velocity in drone's forward direction
    vy: float = 0.0  # Desired velocity in drone's right direction
    vz: float = 0.0  # Desired vertical velocity
    yaw_rate: float = 0.0  # Desired yaw rate (rad/s)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def update_dynamics(
    state: DroneState,
    cmd: DroneCommand,
    config: SimConfig,
) -> DroneState:
    """
    Update drone state based on command using simple Euler integration.
    
    Commands are in body frame (forward/right/up).
    State velocities are in world frame.
    """
    dt = config.dt
    new_state = state.copy()

    # Convert body-frame velocities to world-frame
    cos_yaw = math.cos(state.yaw)
    sin_yaw = math.sin(state.yaw)

    # Target world velocities from command
    target_vx = cmd.vx * cos_yaw - cmd.vy * sin_yaw
    target_vy = cmd.vx * sin_yaw + cmd.vy * cos_yaw
    target_vz = cmd.vz

    # Clamp target velocities
    speed = math.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
    if speed > config.max_speed:
        scale = config.max_speed / speed
        target_vx *= scale
        target_vy *= scale
        target_vz *= scale

    # Apply acceleration limits (simple approach: limit velocity change per tick)
    max_dv = config.max_accel * dt

    new_state.vx = state.vx + _clamp(target_vx - state.vx, -max_dv, max_dv)
    new_state.vy = state.vy + _clamp(target_vy - state.vy, -max_dv, max_dv)
    new_state.vz = state.vz + _clamp(target_vz - state.vz, -max_dv, max_dv)

    # Clamp yaw rate
    new_state.yaw_rate = _clamp(cmd.yaw_rate, -config.max_yaw_rate, config.max_yaw_rate)

    # Integrate position
    new_state.x = state.x + new_state.vx * dt
    new_state.y = state.y + new_state.vy * dt
    new_state.z = state.z + new_state.vz * dt

    # Integrate yaw
    new_state.yaw = _normalize_angle(state.yaw + new_state.yaw_rate * dt)

    # Clamp z to ground
    if new_state.z < 0:
        new_state.z = 0
        new_state.vz = 0

    return new_state
