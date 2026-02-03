"""Course, gate, and obstacle representation."""

from dataclasses import dataclass, field
import math

from .drone import DroneState
from .config import SimConfig


@dataclass
class Gate:
    """A gate/waypoint the drone must pass through."""

    x: float
    y: float
    z: float = 0.0
    radius: float = 1.5  # Pass-through radius (meters)
    yaw: float = 0.0  # Gate orientation (optional, for future use)

    def distance_to(self, state: DroneState) -> float:
        """Calculate distance from drone to gate center."""
        dx = self.x - state.x
        dy = self.y - state.y
        dz = self.z - state.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def bearing_from(self, state: DroneState) -> float:
        """
        Calculate bearing angle from drone to gate.
        
        Returns angle in radians relative to drone's yaw.
        Positive = gate is to the right, Negative = gate is to the left.
        """
        dx = self.x - state.x
        dy = self.y - state.y
        world_angle = math.atan2(dy, dx)
        relative_angle = world_angle - state.yaw

        # Normalize to [-π, π]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        return relative_angle

    def elevation_from(self, state: DroneState) -> float:
        """
        Calculate elevation angle from drone to gate.
        
        Returns angle in radians.
        Positive = gate is above, Negative = gate is below.
        """
        dx = self.x - state.x
        dy = self.y - state.y
        dz = self.z - state.z
        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        
        if horizontal_dist < 0.01:
            # Directly above/below
            return math.copysign(math.pi / 2, dz)
        
        return math.atan2(dz, horizontal_dist)


@dataclass
class Obstacle:
    """A static obstacle the drone must avoid."""

    x: float
    y: float
    z: float = 0.0
    radius: float = 1.0  # Collision radius (meters)

    def distance_to(self, state: DroneState) -> float:
        """Calculate distance from drone to obstacle center."""
        dx = self.x - state.x
        dy = self.y - state.y
        dz = self.z - state.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def bearing_from(self, state: DroneState) -> float:
        """Calculate bearing angle from drone to obstacle."""
        dx = self.x - state.x
        dy = self.y - state.y
        world_angle = math.atan2(dy, dx)
        relative_angle = world_angle - state.yaw

        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        return relative_angle

    def is_collision(self, state: DroneState, drone_radius: float = 0.3) -> bool:
        """Check if drone has collided with this obstacle."""
        return self.distance_to(state) <= (self.radius + drone_radius)


@dataclass
class Course:
    """A sequence of gates forming a course, with optional obstacles."""

    gates: list[Gate] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    current_gate_idx: int = 0
    gates_missed: int = 0
    collisions: int = 0

    def get_current_gate(self) -> Gate | None:
        """Get the current target gate, or None if course complete."""
        if self.current_gate_idx >= len(self.gates):
            return None
        return self.gates[self.current_gate_idx]

    def check_gate_passed(self, state: DroneState) -> bool:
        """Check if drone has passed the current gate."""
        gate = self.get_current_gate()
        if gate is None:
            return False

        distance = gate.distance_to(state)
        if distance <= gate.radius:
            return True
        return False

    def check_collision(self, state: DroneState) -> bool:
        """Check if drone has collided with any obstacle."""
        for obstacle in self.obstacles:
            if obstacle.is_collision(state):
                self.collisions += 1
                return True
        return False

    def get_nearest_obstacle(self, state: DroneState) -> tuple[Obstacle | None, float]:
        """Get the nearest obstacle and its distance."""
        if not self.obstacles:
            return None, float("inf")

        nearest = None
        min_dist = float("inf")
        for obs in self.obstacles:
            dist = obs.distance_to(state)
            if dist < min_dist:
                min_dist = dist
                nearest = obs

        return nearest, min_dist

    def advance_gate(self) -> None:
        """Move to the next gate."""
        self.current_gate_idx += 1

    def is_complete(self) -> bool:
        """Check if all gates have been passed."""
        return self.current_gate_idx >= len(self.gates)

    def gates_passed(self) -> int:
        """Number of gates successfully passed."""
        return self.current_gate_idx

    def gates_total(self) -> int:
        """Total number of gates in course."""
        return len(self.gates)


def is_out_of_bounds(state: DroneState, config: SimConfig) -> bool:
    """Check if drone is outside the world bounds."""
    if state.x < config.bounds_min[0] or state.x > config.bounds_max[0]:
        return True
    if state.y < config.bounds_min[1] or state.y > config.bounds_max[1]:
        return True
    if state.z < config.bounds_min[2] or state.z > config.bounds_max[2]:
        return True
    return False
