"""3D Matplotlib visualizer for the drone simulator."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .drone import DroneState
from .world import Course


class DroneVisualizer:
    """Real-time 3D visualizer using Matplotlib."""

    def __init__(self, course: Course, title: str = "Maria1 Drone Sim"):
        plt.ion()  # Interative mode on
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        
        self.course = course
        self.history_x = []
        self.history_y = []
        self.history_z = []

        # Setup plot elements
        self.path_line, = self.ax.plot([], [], [], 'b-', alpha=0.5, label="Path")
        self.drone_marker, = self.ax.plot([], [], [], 'ro', markersize=8, label="Drone")
        self.front_line, = self.ax.plot([], [], [], 'r-', linewidth=2)
        
        # Draw gates
        self._draw_course()
        
        # Set persistent axis limits
        self._set_limits()
        
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_course(self):
        """Draw all gates and obstacles in the course."""
        # Draw gates
        for i, gate in enumerate(self.course.gates):
            self.ax.scatter(gate.x, gate.y, gate.z, color='green', s=100, marker='s')
            self.ax.text(gate.x, gate.y, gate.z + 1, f"Gate {i+1}", color='green')

        # Draw obstacles
        for i, obs in enumerate(self.course.obstacles):
            # Draw obstacle as red circle
            self.ax.scatter(obs.x, obs.y, obs.z, color='red', s=200, marker='o', alpha=0.7)
            # Draw danger radius
            theta = np.linspace(0, 2 * np.pi, 20)
            circle_x = obs.x + obs.radius * np.cos(theta)
            circle_y = obs.y + obs.radius * np.sin(theta)
            circle_z = np.full_like(theta, obs.z)
            self.ax.plot(circle_x, circle_y, circle_z, 'r-', alpha=0.5)

    def _set_limits(self):
        """Initialize axis limits based on course and expected movement."""
        all_x = [g.x for g in self.course.gates] + [o.x for o in self.course.obstacles] + [0]
        all_y = [g.y for g in self.course.gates] + [o.y for o in self.course.obstacles] + [0]
        all_z = [g.z for g in self.course.gates] + [o.z for o in self.course.obstacles] + [0, 5]
        
        margin = 5
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        self.ax.set_zlim(min(all_z) - 2, max(all_z) + margin)
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')

    def update(self, state: DroneState):
        """Update the visualizer with the latest drone state."""
        self.history_x.append(state.x)
        self.history_y.append(state.y)
        self.history_z.append(state.z)
        
        # Update path
        self.path_line.set_data(self.history_x, self.history_y)
        self.path_line.set_3d_properties(self.history_z)
        
        # Update drone position
        self.drone_marker.set_data([state.x], [state.y])
        self.drone_marker.set_3d_properties([state.z])
        
        # Update drone facing direction
        forward_len = 1.5
        fx = state.x + forward_len * np.cos(state.yaw)
        fy = state.y + forward_len * np.sin(state.yaw)
        fz = state.z
        self.front_line.set_data([state.x, fx], [state.y, fy])
        self.front_line.set_3d_properties([state.z, fz])
        
        # Keep window responsive
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def is_open(self) -> bool:
        """Check if the figure window is still open."""
        return plt.fignum_exists(self.fig.number)

    def close(self):
        """Close the visualizer window."""
        plt.close(self.fig)
