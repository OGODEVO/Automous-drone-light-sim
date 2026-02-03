# Maria Physics - MuJoCo Drone Simulator

Real physics simulation for autonomous drone control using DeepMind's MuJoCo.

## Overview

This project builds on `maria1` (kinematic simulator) by adding **true physics**:
- Gravity, inertia, and collisions
- Individual motor thrust simulation
- PID cascade control (Velocity → Attitude → Rate → Motors)

## Quick Start

```bash
# Install dependencies
pip install mujoco

# Run headless (no GUI)
python main.py

# Run with 3D viewer (macOS)
mjpython main.py --gui
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  BRAIN (FSM)                                                │
│  "Fly to gate at 2 m/s"                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓ vx, vy, vz
┌─────────────────────────────────────────────────────────────┐
│  VELOCITY CONTROLLER                                        │
│  vx_error → pitch_target                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓ roll, pitch targets
┌─────────────────────────────────────────────────────────────┐
│  ATTITUDE CONTROLLER                                        │
│  angle_error → rate_target                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓ p, q, r targets
┌─────────────────────────────────────────────────────────────┐
│  RATE CONTROLLER                                            │
│  rate_error → motor_moments                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓ motor 1-4 thrust
┌─────────────────────────────────────────────────────────────┐
│  MuJoCo PHYSICS                                             │
│  Forces → Acceleration → Velocity → Position                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
maria_physics/
├── main.py              # Simulation loop
├── controller.py        # PID cascade controller
├── models/
│   └── quadrotor.xml    # MuJoCo drone model (MJCF)
└── pyproject.toml
```

## Current Status

| Feature | Status |
|---------|--------|
| Hover (z hold) | ✅ Working |
| Attitude stabilization | ✅ Working |
| Velocity control | 🔧 Tuning in progress |
| FSM brain integration | ⏳ Pending |

## Motor Layout (+ Configuration)

```
       Motor 1 (+X, Front, Red)
              ↑
              |
Motor 3 ←-----+-----→ Motor 4
(+Y, Blue)    |       (-Y, Yellow)
              |
              ↓
       Motor 2 (-X, Back, Green)
```

## Next Steps

1. Fix velocity controller sign (currently inverted)
2. Tune PID gains for stable forward flight
3. Port FSM brain from `maria1`
4. Add obstacle course
