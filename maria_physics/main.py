"""
MuJoCo Drone Demo with PID Control

Usage:
  Headless:  python main.py
  With GUI:  mjpython main.py --gui   (macOS requires mjpython)
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import sys
import time

from controller import DroneController


def quat2euler(quat):
    """Convert Quaternion [w, x, y, z] to Euler [roll, pitch, yaw]."""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_state(data, drone_body_id):
    """Extract drone state from MuJoCo data."""
    # Position and Velocity
    pos = data.body(drone_body_id).xpos
    vel = data.body(drone_body_id).cvel[3:]  # Linear velocity
    
    # Orientation (Quaternion) and Angular Velocity
    quat = data.body(drone_body_id).xquat
    roll, pitch, yaw = quat2euler(quat)
    
    # Body rates (p, q, r) - simplified from global gyro
    # In a real drone, the gyro measures body rates directly.
    # Here we approximate from cvel angular part.
    gyro = data.body(drone_body_id).cvel[:3]
    
    return {
        'x': pos[0], 'y': pos[1], 'z': pos[2],
        'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
        'roll': roll, 'pitch': pitch, 'yaw': yaw,
        'p': gyro[0], 'q': gyro[1], 'r': gyro[2]
    }


def run_simulation(model, data, drone_body_id, gui=False):
    """Run simulation loop with velocity controller."""
    controller = DroneController(model)
    
    # Target: Fly forward at 2 m/s while holding altitude at z=2.0m
    vx_target = 2.0  # m/s forward
    vy_target = 0.0
    z_target = 2.0
    
    print("\n=== VELOCITY CONTROL TEST ===")
    print(f"Target: vx={vx_target} m/s, Hold z={z_target}m")
    
    viewer = None
    if gui:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    start_time = time.time()
    log_count = 0
    
    while data.time < 10.0:
        # 1. Read Sensors
        state = get_state(data, drone_body_id)
        
        # 2. Run Velocity Controller
        dt = model.opt.timestep
        motors = controller.compute_from_velocity(
            state,
            vx_target=vx_target,
            vy_target=vy_target,
            z_target=z_target,
            dt=dt
        )
        
        # 3. Apply to Motors
        data.ctrl[:] = motors
        
        # 4. Step Physics
        mujoco.mj_step(model, data)
        
        # GUI / Logging
        if gui and viewer.is_running():
            viewer.sync()
            
        if int(data.time * 10) > log_count:
            log_count += 1
            print(f"t={data.time:.1f}s  x={state['x']:.1f}m  vx={state['vx']:.2f}m/s  z={state['z']:.2f}m  pitch={np.degrees(state['pitch']):.1f}°")
            
    if viewer:
        viewer.close()
        
    print(f"\nFinal: x={state['x']:.1f}m  vx={state['vx']:.2f}m/s")


def main():
    # Load model
    model_path = Path(__file__).parent / "models" / "quadrotor.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    
    gui = "--gui" in sys.argv
    run_simulation(model, data, drone_body_id, gui=gui)


if __name__ == "__main__":
    main()
