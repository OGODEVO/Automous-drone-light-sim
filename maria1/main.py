"""
Maria1 - Light Drone Simulator

3D Flight Drill: Navigate gates at varying altitudes.
"""

from pathlib import Path
from datetime import datetime

from sim import SimConfig, DroneState, Course, Gate, Obstacle, run_simulation, TickLogger, DroneVisualizer
from perception import create_perception
from brain import DroneBrain


def main():
    """Run 3D flight course with altitude changes."""
    print("=" * 50)
    print("MARIA1 - 3D FLIGHT DRILL")
    print("Gates at varying altitudes (Z-axis)")
    print("=" * 50)

    # 3D flight config: 20% dropout, moderate speed
    config = SimConfig(
        hz=30,
        noise_dropout=0.20,
        noise_jitter=0.1,
        noise_delay_ticks=1,
        max_time=45.0,
    )

    # 3D course: gates at different heights
    course = Course(
        gates=[
            Gate(x=30, y=0, z=5, radius=2.5),     # Climb to 5m
            Gate(x=60, y=15, z=10, radius=2.5),   # Climb + curve right
            Gate(x=90, y=0, z=3, radius=2.5),     # Dive down
            Gate(x=120, y=-10, z=8, radius=2.5),  # Climb + curve left
            Gate(x=150, y=0, z=0, radius=2.5),    # Return to ground level
        ],
        obstacles=[
            # Mid-air obstacle
            Obstacle(x=45, y=5, z=7, radius=2.0),
        ],
    )

    # Initial state: at origin, altitude 2m (slightly elevated)
    initial_state = DroneState(x=0, y=0, z=2, yaw=0)

    # Create perception with noise
    perception = create_perception(config)

    # Create brain
    brain = DroneBrain()

    # Setup logging
    log_dir = Path(__file__).parent / "logs"
    logger = TickLogger(log_dir)

    # Setup visualization
    viz = DroneVisualizer(course)

    print(f"\nConfig:")
    print(f"  Hz: {config.hz}")
    print(f"  Dropout: {config.noise_dropout * 100:.0f}%")
    print(f"  Max time: {config.max_time}s")
    print(f"\n3D Course: {course.gates_total()} gates, {len(course.obstacles)} obstacles")
    for i, gate in enumerate(course.gates):
        print(f"  Gate {i+1}: ({gate.x}, {gate.y}, z={gate.z})")
    for i, obs in enumerate(course.obstacles):
        print(f"  Obstacle {i+1}: ({obs.x}, {obs.y}, z={obs.z}) r={obs.radius}")

    print("\nRunning simulation...")
    print("-" * 50)

    # Run simulation
    result, final_state = run_simulation(
        config=config,
        course=course,
        brain=brain.step,
        perception=perception,
        initial_state=initial_state,
        logger=logger,
        visualizer=viz,
        realtime=True,
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Completed: {result.completed}")
    print(f"  Time: {result.time_elapsed:.2f}s")
    print(f"  Gates: {result.gates_passed}/{result.gates_total}")
    print(f"  Collisions: {result.collisions}")
    print(f"  Termination: {result.termination_reason}")
    print(f"\nFinal position: ({final_state.x:.1f}, {final_state.y:.1f}, z={final_state.z:.1f})")

    # Save log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"run_{timestamp}.json"
    log_path = logger.save(log_file)
    if log_path:
        print(f"\nLog saved: {log_path}")

    return result.completed and result.collisions == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
