from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import numpy.typing as npt

from robot_state import RobotState
from trajectory import Trajectory
from kinematic_utils import forward_kinematics_3d, generate_circle_3d, get_arm_states_from_cartesian_waypoints_3d


def plot_manipulator(manipulator: Chain, joint_angles: list[float], ax: Axes) -> None:
    link_positions = forward_kinematics_3d(manipulator, [0] + list(joint_angles) + [0])
    ax.plot(*zip(*link_positions), "o-", lw=2)


def animate_trajectory(
    arm: Chain,
    trajectory: Trajectory,
    fps: int,
    cartesian_waypoints: list[list[float]],
) -> None:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("3-DOF Arm")
    ax.scatter(*zip(*cartesian_waypoints), color="red", s=10)

    initial_arm_state = trajectory.position_at(0).values()
    initial_link_positions = forward_kinematics_3d(arm, [0] + list(initial_arm_state[:3]) + [0])

    (line,) = ax.plot([], [], [], "o-", lw=2)

    def init_animation():
        nonlocal initial_link_positions
        x, y, z = zip(*initial_link_positions)
        line.set_data(x, y)
        line.set_3d_properties(z)
        return (line,)

    def update_animation(frame: int):
        nonlocal animation
        traj_time = frame / fps
        if traj_time > trajectory.total_time():
            traj_time = trajectory.total_time()

        arm_state = trajectory.position_at(traj_time).values()
        link_positions = forward_kinematics_3d(arm, [0] + list(arm_state[:3]) + [0])
        x, y, z = zip(*link_positions)
        line.set_data(x, y)  # Update the line segments
        line.set_3d_properties(z)

        return (line,)

    interval = 1000 / fps
    total_frames_count = int(trajectory.total_time() * fps)
    animation = FuncAnimation(
        fig, update_animation, init_func=init_animation, frames=total_frames_count, interval=interval, blit=True
    )
    #animation.save("3-dof-cart.gif", writer=PillowWriter(fps=20))

    plt.show()


link1 = 2.0
link2 = 2.0
link3 = 2.0
max_velocity = 1.5
max_acceleration = 1

if __name__ == "__main__":

    arm = Chain(
        name="3d_robot",
        links=[
            OriginLink(),
            URDFLink(
                name="link1",
                origin_translation=np.array([0, 0, 0]),
                origin_orientation=np.array([0, 0, 0]),
                rotation=np.array([0, 0, 1]),
            ),
            URDFLink(
                name="link2",
                origin_translation=np.array([0, 0, link1]),
                origin_orientation=np.array([np.pi / 2, -np.pi / 2, 0]),
                rotation=np.array([0, 0, 1]),
            ),
            URDFLink(
                name="link3",
                origin_translation=np.array([link2, 0, 0]),
                origin_orientation=np.array([0, 0, 0]),
                rotation=np.array([0, 0, 1]),
            ),
            URDFLink(
                name="end_effector",
                origin_translation=np.array([link3, 0, 0]),
                origin_orientation=np.array([0, 0, 0]),
                rotation=np.array([0, 0, 1]),
            ),
        ],
    )

    cartesian_points = generate_circle_3d(radius=1.5, points_count=20)
    cartesian_points
    transformation = np.eye(4)
    alpha = np.pi / 3
    transformation[:-1, :-1] = np.array(
        [[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]]
    )
    transformation[:-1, 3:] = np.array([[2], [-1], [3]])
    cartesian_points = np.dot(transformation[:-1, :-1], cartesian_points.T).T + transformation[:-1, 3]
    arm_states = get_arm_states_from_cartesian_waypoints_3d(arm, cartesian_points)
    trajectory = Trajectory(arm_states, max_vel=max_velocity, max_acc=max_acceleration)
    animate_trajectory(arm, trajectory, fps=20, cartesian_waypoints=cartesian_points)
