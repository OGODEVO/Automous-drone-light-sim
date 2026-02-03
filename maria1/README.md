# Maria1 Drone Simulator 🚁

Maria1 is a lightweight, deterministic drone simulator designed for rapid prototyping of autonomous flight logic. It focuses on Finite State Machine (FSM) control and handling noisy perception data (dropout, jitter, delay) at high speeds.

## Features

- **High-Speed Autonomous Flight**: Optimized for racing at speeds up to **26 m/s**.
- **Robust Perception Handling**: Stubbed perception system that simulates real-world sensor dropout and jitter.
- **FSM Brain**: Multi-state controller (`SEARCH`, `TRACK`, `AVOID`, `RECOVER`) with stability-first design.
- **Collision Detection**: Real-time obstacle avoidance and collision monitoring.
- **3D Visualization**: Live 3D path and heading visualization using Matplotlib.
- **Deterministic Logging**: Detailed JSON logging for every tick, enabling post-flight analysis.

## Project Structure

- `main.py`: Entry point for running simulation drills.
- `brain/`: FSM logic and flight controller.
- `sim/`: Core physics, world representation (gates/obstacles), and visualization.
- `perception/`: Simulated sensor stub with configurable noise.
- `scoring/`: Metrics and performance evaluation.
- `logs/`: (Ignored) Auto-generated simulation logs.

## Quick Start

### 1. Install Dependencies
```bash
pip install matplotlib numpy
```

### 2. Run Simulation
```bash
python main.py
```

## Simulation Configuration

Parameters are tuned in `main.py` via `SimConfig`:
- `hz`: Control frequency (default 30 Hz).
- `noise_dropout`: Probability of blindness (e.g., 0.25 = 25% blind).
- `noise_jitter`: Measurement noise in meters.
- `noise_delay_ticks`: Sensor lag.

## Goals & Drills

- **Stress Test**: 50% sensor dropout survival.
- **Obstacle Course**: Navigation through static pillars.
- **High-Speed Drill**: 26 m/s through a 200m track.
