# Drone Engineering Basics: A Developer's Guide

This guide explains how autonomous drones "think," "see," and "act," mapping real-world concepts to our simulator implementation.

---

## 1. How a Drone Knows "Where Am I?" (State Estimation)

A drone needs to know its position `(x, y, z)`, orientation `(yaw)`, and velocity `(vx, vy, vz)`.

### Real World
Drones fuse data from multiple sensors using a **Kalman Filter**:
- **GPS**: Global position (accurate to ~2m).
- **IMU (Accelerometer/Gyro)**: Fast updates on tilt and acceleration (1000Hz).
- **Barometer**: Altitude (Z-axis).
- **Optical Flow**: Downward camera tracking texture to execute "hover in place".

### In Our Sim (`sim/drone.py`)
We "cheat" and provide the perfect truth from the physics engine:
```python
@dataclass
class DroneState:
    x: float; y: float; z: float  # precise position
    yaw: float                    # heading
```

---

## 2. How a Drone Knows "What's Around Me?" (Perception)

Drones build a 3D map of the world to find goals and avoid obstacles.

### Real World Sensors
1.  **VIO (Visual Inertial Odometry)**: Cameras tracking feature points to build a sparse map.
2.  **Lidar**: Shooting lasers to measure precise distance (50-100m range).
3.  **Depth Cameras (RealSense)**: Stereo vision to see depth up to 10m.

### The "Cone of Safety"
Everything has a range limit.
- **Field of View (FOV)**: Can the camera see 60° wide or 120° wide?
- **Max Range**: How far before pixels get too blurry?

### In Our Sim (`perception/stub.py`)
We simulate these limits:
- `obstacle_detection_range = 15.0`: Beyond 15m, the drone is "blind" to obstacles.
- `obstacle_bearing`: Angle to the object.
- `gate_elevation`: Pitch angle to the target.

---

## 3. How a Drone Decides "What to Do?" (The Brain)

This is the code you write. It typically runs in a **Control Loop** at 30Hz - 100Hz.

### The Loop
1.  **Sense**: Get latest `PerceptionResult`.
2.  **Plan**: Update state machine (`SEARCH` -> `TRACK`).
3.  **Act**: Output velocity commands (`DroneCommand`).

### Finite State Machine (FSM)
A common simple brain structure:
1.  **SEARCH**: "I don't see the gate. Spin around."
2.  **TRACK**: "I see it! Fly toward it."
3.  **AVOID**: "Something is in my way! Steer left!"
4.  **RECOVER**: "I've been blind too long. Back up and re-orient."

### In Our Sim (`brain/fsm.py`)
```python
def Step(obs):
    if threat_detected: return AVOID
    if gate_visible: return TRACK
    if lost_signal: return RECOVER
    return SEARCH
```

---

## 4. How a Drone Moves (Control Theory)

Once the brain says "Go forward at 5 m/s", the lower-level controller makes it happen.

### PID Controller (Proportional-Integral-Derivative)
This math keeps the drone stable.
- **P (Proportional)**: "Target is 10m away? Push stick 10%."
- **I (Integral)**: "Wind is pushing me back? Push harder over time."
- **D (Derivative)**: "Approaching target too fast? Slow down to avoid overshooting."

### Our Sim Implementation
We use simplified **P-Controllers**:
```python
# Proportional Pitch Control
error = target_elevation - current_pitch
vz = kp * error  # kp = 8.0 (The gain we tuned!)
```
- **If `kp` is too low**: Drone reacts too slowly (logs showed `vz=0.3` when we needed `5.0`).
- **If `kp` is too high**: Drone oscillates (wobbles) aggressively.

---

## 5. The "Latency" Killer

Robotics is a fight against time.
1.  **Sensor Lag**: Camera takes 30ms to capture + 50ms to process AI. The drone acts on data from 80ms ago.
2.  **Physics Inertia**: Motors take time to spin up. The drone slides before turning (drifting).

**Lesson from our 3D Drill**:
We had to increase `avoid_distance` because by the time the drone *shifted weight* to turn, it had already traveled 10 meters!
