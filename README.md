# Autonomous Drone Light Simulator

A Python-based drone simulation platform for learning autonomous flight control — from basic FSM logic to real physics with MuJoCo.

## Projects

| Folder | Description | Complexity |
|--------|-------------|------------|
| [`maria1/`](./maria1/) | Kinematic simulator with FSM brain, fake sensors, and 3D visualization | Beginner |
| [`maria_physics/`](./maria_physics/) | MuJoCo-based physics with PID cascade controller | Intermediate |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/OGODEVO/Automous-drone-light-sim.git
cd Automous-drone-light-sim

# Run kinematic sim (no physics, just logic)
cd maria1
pip install matplotlib numpy
python main.py

# Run physics sim (real forces and torques)
cd ../maria_physics
pip install mujoco
python main.py              # Headless
mjpython main.py --gui      # With 3D viewer (macOS)
```

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                        BRAIN (FSM)                         │
│              SEARCH → TRACK → AVOID → RECOVER              │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│                    CONTROLLER (PID)                        │
│         Velocity → Attitude → Rate → Motor Mixing          │
└────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────┬───────────────────────────────────┐
│      maria1 (Fake)    │       maria_physics (Real)        │
│   Instant velocity    │   MuJoCo forces & torques         │
└───────────────────────┴───────────────────────────────────┘
```

## Learning Path

1. **Start with `maria1`** — Understand FSM states, perception, and gate tracking
2. **Move to `maria_physics`** — Learn PID control and motor mixing
3. **Tune the gains** — Experience why real drones are hard to fly
4. **Port the brain** — Connect FSM to physics controller

## Requirements

- Python 3.10+
- `numpy`, `matplotlib` (maria1)
- `mujoco` (maria_physics)

## License

MIT
