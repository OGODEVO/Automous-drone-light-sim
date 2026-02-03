"""
Quadrotor Controller (PID)

This module translates high-level commands (Roll/Pitch/Yaw rates, Thrust) 
into low-level motor commands (RPM/Force).

Architecture:
1. Attitude Controller (Rate PIDs):
   - Input: Desired body rates (p, q, r)
   - Output: Motor mixing (moments)

2. Position/Velocity Controller (Outer Loop):
   - Input: Desired Velocity (vx, vy, vz)
   - Output: Desired Roll/Pitch Angles
"""

import numpy as np
import mujoco
from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    limit: float = 100.0  # Output limit


class PID:
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        # P term
        p = self.cfg.kp * error

        # I term
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.cfg.limit, self.cfg.limit)
        i = self.cfg.ki * self.integral

        # D term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d = self.cfg.kd * derivative
        self.prev_error = error

        return p + i + d


class DroneController:
    """
    Full cascade controller for quadrotor:
    
    Velocity → Attitude → Rate → Motor Mixing
    
    Usage:
        motors = controller.compute_from_velocity(state, vx=5, vy=0, vz=0, yaw_rate=0)
    """
    
    def __init__(self, model: mujoco.MjModel):
        # Physical parameters
        self.mass = 0.780  # kg
        self.gravity = 9.81  # m/s^2
        self.hover_thrust = (self.mass * self.gravity) / 4.0
        
        # Velocity limits (safety)
        self.max_tilt = 0.25  # ~15 degrees max tilt (was 0.5)

        # === VELOCITY PIDs (Outermost loop) ===
        # Input: Velocity error (m/s)
        # Output: Desired tilt angle (radians)
        # Lower gains for smoother, safer flight
        self.vx_pid = PID(PIDConfig(kp=0.08, ki=0.005, kd=0.02, limit=self.max_tilt))
        self.vy_pid = PID(PIDConfig(kp=0.08, ki=0.005, kd=0.02, limit=self.max_tilt))
        self.vz_pid = PID(PIDConfig(kp=5.0, ki=1.0, kd=2.0))  # For altitude rate control

        # === ATTITUDE PIDs (Middle loop) ===
        # Input: Angle error (radians)
        # Output: Desired angular rate (rad/s)
        self.roll_angle_pid = PID(PIDConfig(kp=6.0, ki=0.0, kd=0.5))
        self.pitch_angle_pid = PID(PIDConfig(kp=6.0, ki=0.0, kd=0.5))

        # === RATE PIDs (Innermost loop) ===
        # Input: Angular rate error (rad/s)
        # Output: Motor moment command
        self.roll_rate_pid = PID(PIDConfig(kp=2.0, ki=0.0, kd=0.1))
        self.pitch_rate_pid = PID(PIDConfig(kp=2.0, ki=0.0, kd=0.1))
        self.yaw_rate_pid = PID(PIDConfig(kp=4.0, ki=0.0, kd=0.1))
        
        # Altitude (position control)
        self.z_pid = PID(PIDConfig(kp=5.0, ki=1.0, kd=2.0))

    def compute_from_velocity(
        self, 
        state: dict, 
        vx_target: float = 0.0,
        vy_target: float = 0.0,
        vz_target: float = 0.0,
        yaw_rate_target: float = 0.0,
        z_target: float | None = None,
        dt: float = 0.002
    ) -> np.ndarray:
        """
        High-level velocity control interface.
        
        Args:
            state: Current drone state (x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r)
            vx_target: Desired forward velocity (m/s) in WORLD frame
            vy_target: Desired lateral velocity (m/s) in WORLD frame
            vz_target: Desired vertical velocity (m/s) - if z_target is None
            yaw_rate_target: Desired yaw rate (rad/s)
            z_target: If set, control altitude position instead of vz
            dt: Timestep
            
        Returns:
            4-element array of motor thrusts
        """
        
        # === 1. VELOCITY → ATTITUDE ===
        # To fly forward (positive vx), we need to pitch down (negative pitch)
        # To fly right (positive vy), we need to roll right (positive roll)
        
        vx_error = vx_target - state['vx']
        vy_error = vy_target - state['vy']
        
        # Note: For + config, positive pitch = forward motion (tilting back motors up)
        pitch_target = self.vx_pid.update(vx_error, dt)
        roll_target = self.vy_pid.update(vy_error, dt)
        
        # Clamp tilt angles for safety
        pitch_target = np.clip(pitch_target, -self.max_tilt, self.max_tilt)
        roll_target = np.clip(roll_target, -self.max_tilt, self.max_tilt)
        
        # === 2. ALTITUDE CONTROL ===
        if z_target is not None:
            # Position control mode
            z_error = z_target - state['z']
            thrust_correction = self.z_pid.update(z_error, dt)
        else:
            # Velocity control mode
            vz_error = vz_target - state['vz']
            thrust_correction = self.vz_pid.update(vz_error, dt)
        
        total_thrust = (self.mass * self.gravity) + thrust_correction
        
        # === 3. ATTITUDE → RATE ===
        roll_error = roll_target - state['roll']
        pitch_error = pitch_target - state['pitch']
        
        target_p = self.roll_angle_pid.update(roll_error, dt)
        target_q = self.pitch_angle_pid.update(pitch_error, dt)
        target_r = yaw_rate_target

        # === 4. RATE → MOMENT ===
        p_error = target_p - state['p']
        q_error = target_q - state['q']
        r_error = target_r - state['r']

        roll_moment = self.roll_rate_pid.update(p_error, dt)
        pitch_moment = self.pitch_rate_pid.update(q_error, dt)
        yaw_moment = self.yaw_rate_pid.update(r_error, dt)

        # === 5. MOTOR MIXING ===
        base = total_thrust / 4.0
        
        m1 = base - roll_moment + pitch_moment - yaw_moment
        m2 = base + roll_moment - pitch_moment - yaw_moment
        m3 = base + roll_moment + pitch_moment + yaw_moment
        m4 = base - roll_moment - pitch_moment + yaw_moment
        
        return np.clip([m1, m2, m3, m4], 0, 10.0)

    def compute_motor_controls(
        self, 
        current_state: dict, 
        target_state: dict, 
        dt: float
    ) -> np.ndarray:
        """
        Legacy interface for angle-based control.
        
        current_state: {roll, pitch, yaw, p, q, r, z, vz}
        target_state: {roll_target, pitch_target, yaw_rate_target, z_target}
        """
        
        # 1. Altitude Control (Z position -> Thrust)
        z_error = target_state['z_target'] - current_state['z']
        thrust_correction = self.z_pid.update(z_error, dt)
        total_thrust = (self.mass * self.gravity) + thrust_correction
        
        # 2. Attitude Control (Angle -> Rate)
        roll_error = target_state['roll_target'] - current_state['roll']
        pitch_error = target_state['pitch_target'] - current_state['pitch']
        
        target_p = self.roll_angle_pid.update(roll_error, dt)
        target_q = self.pitch_angle_pid.update(pitch_error, dt)
        target_r = target_state['yaw_rate_target']

        # 3. Rate Control (Rate -> Moment)
        p_error = target_p - current_state['p']
        q_error = target_q - current_state['q']
        r_error = target_r - current_state['r']

        roll_moment = self.roll_rate_pid.update(p_error, dt)
        pitch_moment = self.pitch_rate_pid.update(q_error, dt)
        yaw_moment = self.yaw_rate_pid.update(r_error, dt)

        # 4. Motor Mixing
        base = total_thrust / 4.0
        
        m1 = base - roll_moment + pitch_moment - yaw_moment
        m2 = base + roll_moment - pitch_moment - yaw_moment
        m3 = base + roll_moment + pitch_moment + yaw_moment
        m4 = base - roll_moment - pitch_moment + yaw_moment
        
        return np.clip([m1, m2, m3, m4], 0, 10.0)
