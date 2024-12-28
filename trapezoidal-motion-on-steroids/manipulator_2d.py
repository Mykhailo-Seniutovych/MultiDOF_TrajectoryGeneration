import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import numpy.typing as npt

from robot_state import RobotState
from trajectory import Trajectory

from kinematic_utils import (
    calculate_positions,
    get_robot_states_from_cartesian_waypoints,
    generate_cartesian_waypoints_on_line,
)


def generate_cartesian_waypoints_on_line(
    line_slope: float, line_intercept: float, x_start: float, x_end: float, points_count: int
) -> npt.NDArray[np.float64]:
    x_values = np.linspace(x_start, x_end, points_count)
    y_values = line_slope * x_values + line_intercept
    waypoints = np.column_stack((x_values, y_values))
    return waypoints


def plot_manipulator(theta1: float, theta2: float, link1: float, link2: float) -> None:
    x1, y1, x2, y2 = calculate_positions(theta1, theta2, link1, link2)

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=1)  # X-axis
    ax.axvline(0, color="black", linewidth=1)  # Y-axis
    ax.grid(True)
    (line,) = ax.plot([0, x1, x2], [0, y1, y2], "o-", lw=2)
    plt.show()


def animate_trajectory(
    trajectory: Trajectory, link1: float, link2: float, fps: int, waypoints: list[RobotState]
) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=1)  # X-axis
    ax.axvline(0, color="black", linewidth=1)  # Y-axis
    ax.grid(True)
    ax.set_title("2-DOF Robot")
    cartesian_waypoints = [
        calculate_positions(waypoint.values()[0], waypoint.values()[1], link1, link2)[-2:] for waypoint in waypoints
    ]
    ax.scatter(*zip(*cartesian_waypoints), color="red", s=10)
    initial_state = trajectory.position_at(0).values()
    x1_initial, y1_initial, x2_initial, y2_initial = calculate_positions(
        initial_state[0], initial_state[1], link1, link2
    )
    (line,) = ax.plot([0, x1_initial, x2_initial], [0, y1_initial, y2_initial], "o-", lw=2)

    # Create the animation

    def update_animation(frame: int):
        traj_time = frame / fps
        state = (
            trajectory.position_at(traj_time).values()
            if traj_time <= trajectory.total_time()
            else trajectory.position_at(trajectory.total_time()).values()
        )
        x1, y1, x2, y2 = calculate_positions(state[0], state[1], link1, link2)
        line.set_data([0, x1, x2], [0, y1, y2])  # Update the line segments
        return (line,)

    interval = 1000 / fps
    total_frames_count = int(trajectory.total_time() * fps)
    animation = FuncAnimation(fig, update_animation, frames=total_frames_count, interval=interval, blit=True)
    animation.save("2-DOF-traj-advanced.gif", writer=PillowWriter(fps=fps))
    plt.show()


link1_length = 2.0
link2_length = 2.0
frame_rate = 60
max_velocity = 1.5  # radians per second
max_acceleration = 1  # radians per second squared

if __name__ == "__main__":
    cartesian_waypoints = generate_cartesian_waypoints_on_line(
        line_slope=-0.1, line_intercept=2.5, x_start=3.3, x_end=-2, points_count=10
    )

    waypoints = get_robot_states_from_cartesian_waypoints(
        cartesian_waypoints, link1_length, link2_length, elbow_up=True
    )
    waypoints = [
        RobotState([np.deg2rad(0), np.deg2rad(0)]),
        RobotState([np.deg2rad(20), np.deg2rad(70)]),
        RobotState([np.deg2rad(100), np.deg2rad(100)]),
        RobotState([np.deg2rad(150), np.deg2rad(150)]),
    ]
    trajectory = Trajectory(waypoints, max_velocity, max_acceleration)
    animate_trajectory(trajectory, link1=link1_length, link2=link2_length, fps=20, waypoints=waypoints)
