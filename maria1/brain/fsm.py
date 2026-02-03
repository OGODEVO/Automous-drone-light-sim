"""Finite State Machine brain for high-speed autonomous navigation."""

from enum import Enum, auto
from dataclasses import dataclass
import math

from perception.stub import PerceptionResult
from sim.drone import DroneCommand


class FSMState(Enum):
    """States for the drone brain FSM."""

    SEARCH = auto()  # Looking for gate
    TRACK = auto()  # Gate visible or recently visible, steering toward it
    AVOID = auto()  # Obstacle detected, steering to avoid
    RECOVER = auto()  # Lost gate for too long, attempting recovery


@dataclass
class BrainConfig:
    """
    Tunable parameters for high-speed autonomous flight.
    
    Key principle: Stability over aggressiveness.
    - High speed when confident (visible, clear path)
    - Conservative when uncertain (blind, recovering)
    """

    # SEARCH mode
    search_yaw_rate: float = 1.5  # rad/s - spin faster at high speed

    # TRACK mode - HIGH SPEED
    track_max_speed: float = 26.0  # m/s - top speed when tracking
    track_min_speed: float = 8.0   # m/s - minimum speed (still fast)
    track_coast_speed: float = 15.0  # m/s - maintain momentum when blind
    track_yaw_gain: float = 2.0  # Increased for tighter carving
    track_pitch_gain: float = 8.0  # Stronger gain for aggressive climbing
    track_max_vz: float = 8.0  # Allow faster vertical moves
    track_approach_gain: float = 0.8  # Speed = distance * gain, capped at max
    yaw_deadband: float = 0.03  # Tighter deadband for precision
    pitch_deadband: float = 0.02  # Ignore small elevation errors

    # AVOID mode (obstacle avoidance) - CONSERVATIVE
    avoid_distance: float = 14.0  # React as soon as detected (was 10.0)
    avoid_critical_distance: float = 5.0  # Emergency at 5m
    avoid_yaw_rate: float = 2.5  # Aggressive turn to clear obstacle
    avoid_speed: float = 10.0  # Still moving but controlled

    # RECOVER mode - CALM
    recover_timeout_ticks: int = 45  # More patience before giving up (1.5s)
    recover_reverse_speed: float = -3.0  # Back up faster but controlled
    recover_coast_speed: float = 8.0  # Forward coast if recently saw gate

    # Thresholds
    lost_threshold_ticks: int = 6  # ~200ms of blindness before concern
    coast_threshold_ticks: int = 3  # ~100ms: coast with confidence


@dataclass
class LastValidState:
    """Memory of last valid observation and command."""
    bearing: float = 0.0
    elevation: float = 0.0
    distance: float = 50.0
    yaw_rate: float = 0.0
    vz: float = 0.0
    speed: float = 0.0
    ticks_since: int = 0


class DroneBrain:
    """
    High-speed FSM brain: SEARCH → TRACK → AVOID → RECOVER
    
    Design principles:
    1. GO FAST when confident (clear visibility, gate in front)
    2. COAST during brief blindness (maintain momentum, don't panic)
    3. SLOW DOWN when uncertain (obstacle near, prolonged blindness)
    4. RECOVER CALMLY from total loss (back up, scan, don't thrash)
    """

    def __init__(self, config: BrainConfig | None = None):
        self.config = config or BrainConfig()
        self.state = FSMState.SEARCH
        self.blind_ticks = 0
        self.visible_ticks = 0
        self.recover_ticks = 0
        self.avoid_ticks = 0
        self.last_valid = LastValidState()

    def reset(self) -> None:
        """Reset brain state."""
        self.state = FSMState.SEARCH
        self.blind_ticks = 0
        self.visible_ticks = 0
        self.recover_ticks = 0
        self.avoid_ticks = 0
        self.last_valid = LastValidState()

    def step(self, obs: PerceptionResult | None) -> tuple[str, DroneCommand]:
        """
        Process observation and return (state_name, command).
        
        obs is None when blind (dropout or no gate).
        """
        cfg = self.config
        is_visible = obs is not None

        # Update visibility counters
        if is_visible:
            self.blind_ticks = 0
            self.visible_ticks += 1
            # Store last valid observation
            self.last_valid.bearing = obs.gate_bearing
            self.last_valid.elevation = obs.gate_elevation
            self.last_valid.distance = obs.gate_distance
            self.last_valid.ticks_since = 0
        else:
            self.blind_ticks += 1
            self.visible_ticks = 0
            self.last_valid.ticks_since += 1

        # Check for obstacle threat
        obstacle_threat = self._check_obstacle_threat(obs, cfg)

        # State transitions
        self._update_state(is_visible, obstacle_threat, cfg)

        # Generate command based on state
        cmd = self._generate_command(obs, is_visible, cfg)

        # Store last valid command (for coasting)
        if is_visible and self.state == FSMState.TRACK:
            self.last_valid.yaw_rate = cmd.yaw_rate
            self.last_valid.vz = cmd.vz
            self.last_valid.speed = cmd.vx

        return self.state.name, cmd

    def _check_obstacle_threat(self, obs: PerceptionResult | None, cfg: BrainConfig) -> bool:
        """Check if there's an obstacle that needs avoiding."""
        if obs is None:
            return False
        if obs.obstacle_distance is None:
            return False

        # Higher speed = need more distance to react
        if obs.obstacle_distance < cfg.avoid_distance:
            if obs.obstacle_bearing is not None and abs(obs.obstacle_bearing) < math.pi / 3:
                return True
        return False

    def _update_state(self, is_visible: bool, obstacle_threat: bool, cfg: BrainConfig) -> None:
        """Handle state transitions with hysteresis."""
        
        # AVOID takes priority when there's an obstacle threat
        if obstacle_threat and self.state in (FSMState.TRACK, FSMState.SEARCH):
            self.state = FSMState.AVOID
            self.avoid_ticks = 0
            return

        if self.state == FSMState.SEARCH:
            if is_visible:
                self.state = FSMState.TRACK

        elif self.state == FSMState.TRACK:
            if self.blind_ticks >= cfg.lost_threshold_ticks:
                self.state = FSMState.RECOVER
                self.recover_ticks = 0

        elif self.state == FSMState.AVOID:
            self.avoid_ticks += 1
            if not obstacle_threat:
                if is_visible:
                    self.state = FSMState.TRACK
                else:
                    self.state = FSMState.SEARCH

        elif self.state == FSMState.RECOVER:
            self.recover_ticks += 1
            if is_visible:
                self.state = FSMState.TRACK
            elif self.recover_ticks >= cfg.recover_timeout_ticks:
                self.state = FSMState.SEARCH

    def _generate_command(
        self, obs: PerceptionResult | None, is_visible: bool, cfg: BrainConfig
    ) -> DroneCommand:
        """Generate movement command based on current state."""

        if self.state == FSMState.SEARCH:
            return self._search_command(cfg)

        elif self.state == FSMState.TRACK:
            if is_visible:
                return self._track_visible_command(obs, cfg)
            else:
                return self._track_blind_command(cfg)

        elif self.state == FSMState.AVOID:
            return self._avoid_command(obs, cfg)

        elif self.state == FSMState.RECOVER:
            return self._recover_command(cfg)

        return DroneCommand()

    def _search_command(self, cfg: BrainConfig) -> DroneCommand:
        """SEARCH: Spin in place looking for gate."""
        return DroneCommand(
            vx=0.0,
            vy=0.0,
            vz=0.0,
            yaw_rate=cfg.search_yaw_rate,
        )

    def _track_visible_command(self, obs: PerceptionResult, cfg: BrainConfig) -> DroneCommand:
        """
        TRACK (visible): Full speed ahead with proportional steering.
        
        Speed scales with distance but is always fast.
        Uses elevation for 3D pitch control.
        """
        bearing = obs.gate_bearing
        distance = obs.gate_distance
        elevation = obs.gate_elevation

        # Proportional yaw steering with deadband
        if abs(bearing) < cfg.yaw_deadband:
            yaw_rate = 0.0
        else:
            yaw_rate = cfg.track_yaw_gain * bearing

        # Proportional pitch control with deadband (3D)
        if abs(elevation) < cfg.pitch_deadband:
            vz = 0.0
        else:
            vz = cfg.track_pitch_gain * elevation
            vz = max(-cfg.track_max_vz, min(cfg.track_max_vz, vz))

        # Speed: fast when far, still fast when close
        # Formula: distance * gain, clamped to [min_speed, max_speed]
        speed = distance * cfg.track_approach_gain
        speed = max(cfg.track_min_speed, min(cfg.track_max_speed, speed))

        return DroneCommand(
            vx=speed,
            vy=0.0,
            vz=vz,
            yaw_rate=yaw_rate,
        )

    def _track_blind_command(self, cfg: BrainConfig) -> DroneCommand:
        """
        TRACK (blind): Coast forward, trust last known trajectory.
        
        Key: Don't panic, don't over-correct. Vision will return.
        """
        # Confidence decays with time blind
        confidence = max(0.3, 1.0 - self.blind_ticks * 0.15)
        
        # Decay yaw rate toward zero (don't keep turning blind)
        decayed_yaw = self.last_valid.yaw_rate * confidence * 0.5
        
        # Decay vz toward zero (don't keep climbing/descending blind)
        decayed_vz = self.last_valid.vz * confidence * 0.5

        # Maintain high speed but slightly reduced
        coast_speed = cfg.track_coast_speed * confidence

        return DroneCommand(
            vx=max(cfg.track_min_speed * 0.5, coast_speed),
            vy=0.0,
            vz=decayed_vz,
            yaw_rate=decayed_yaw,
        )

    def _avoid_command(self, obs: PerceptionResult | None, cfg: BrainConfig) -> DroneCommand:
        """
        AVOID: Steer yaw away from obstacle, but maintain pitch toward gate.
        
        This allows the drone to fly OVER or UNDER obstacles while dodging.
        Safety: When close to obstacle, prefer climbing over descending.
        """
        # Default fallback if blind
        if obs is None or obs.obstacle_bearing is None:
            # Use last known vz to maintain altitude trajectory
            decayed_vz = self.last_valid.vz * 0.5
            return DroneCommand(
                vx=cfg.avoid_speed * 0.5,
                vy=0.0,
                vz=decayed_vz,
                yaw_rate=0.0,
            )

        # Turn AWAY from the obstacle (yaw)
        avoid_direction = -math.copysign(1.0, obs.obstacle_bearing)
        
        # More aggressive when closer
        urgency = 1.0
        is_critical = False
        if obs.obstacle_distance is not None and obs.obstacle_distance < cfg.avoid_critical_distance:
            urgency = 1.5
            is_critical = True

        yaw_rate = avoid_direction * cfg.avoid_yaw_rate * urgency

        # Slow down proportionally to threat level
        speed = cfg.avoid_speed
        if is_critical:
            speed = cfg.avoid_speed * 0.6

        # KEEP climbing/descending toward the gate (3D fix)
        # Use gate elevation if visible, otherwise decay last known
        if obs.gate_elevation is not None:
            vz = cfg.track_pitch_gain * obs.gate_elevation
            vz = max(-cfg.track_max_vz, min(cfg.track_max_vz, vz))
        else:
            vz = self.last_valid.vz * 0.7

        # SAFETY: When obstacle is dangerously close, DON'T descend
        # This prevents noisy elevation readings from causing us to dive into obstacles
        if is_critical and vz < 0:
            vz = max(0.0, self.last_valid.vz * 0.5)  # Coast level or climb

        return DroneCommand(
            vx=speed,
            vy=0.0,
            vz=vz,
            yaw_rate=yaw_rate,
        )

    def _recover_command(self, cfg: BrainConfig) -> DroneCommand:
        """
        RECOVER: Calmly back up and scan for gate.
        
        Don't thrash. Systematic search.
        """
        # If we just lost the gate, coast forward briefly
        if self.last_valid.ticks_since < 15:
            return DroneCommand(
                vx=cfg.recover_coast_speed,
                vy=0.0,
                vz=0.0,
                yaw_rate=math.copysign(cfg.search_yaw_rate * 0.3, self.last_valid.bearing),
            )

        # Otherwise back up and scan
        scan_direction = math.copysign(1.0, self.last_valid.bearing)

        return DroneCommand(
            vx=cfg.recover_reverse_speed,
            vy=0.0,
            vz=0.0,
            yaw_rate=cfg.search_yaw_rate * 0.7 * scan_direction,
        )
