import numpy as np
import numpy.typing as npt

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from robot_state import RobotState


#############################
# Simple 2 DOF planar robot #
#############################
def calculate_positions(theta1: float, theta2: float, link1: float, link2: float) -> tuple[float, float, float, float]:
    x1 = link1 * np.cos(theta1)
    y1 = link1 * np.sin(theta1)
    x2 = x1 + link2 * np.cos(theta1 + theta2)
    y2 = y1 + link2 * np.sin(theta1 + theta2)
    return (x1, y1, x2, y2)


def calculate_angles(x: float, y: float, link1: float, link2: float) -> tuple[float, float, float, float]:
    radius = np.sqrt(x**2 + y**2)
    gamma = np.arccos((link1**2 + link2**2 - radius**2) / (2 * link1 * link2))
    assert np.isnan(gamma) == False

    theta2_elbow_down = np.pi - gamma
    theta2_elbow_up = gamma - np.pi
    beta = np.arctan2(y, x)
    alpha = np.arccos((radius**2 + link1**2 - link2**2) / (2 * radius * link1))
    theta1_elbow_down = beta - alpha
    theta1_elbow_up = beta + alpha
    return (theta1_elbow_down, theta2_elbow_down, theta1_elbow_up, theta2_elbow_up)


def get_robot_states_from_cartesian_waypoints(
    cartesian_waypoints: list[list[float]], link1: float, link2: float, elbow_up: bool = False
) -> list[RobotState]:
    robot_states: list[RobotState] = []
    for x, y in cartesian_waypoints:
        theta1_down, theta2_down, theta1_up, theta2_up = calculate_angles(x, y, link1, link2)
        theta1, theta2 = (theta1_up, theta2_up) if elbow_up else (theta1_down, theta2_down)
        robot_states.append(RobotState([theta1, theta2]))
    return robot_states


def generate_cartesian_waypoints_on_line(
    line_slope: float, line_intercept: float, x_start: float, x_end: float, points_count: int
) -> npt.NDArray[np.float64]:
    x_values = np.linspace(x_start, x_end, points_count)
    y_values = line_slope * x_values + line_intercept
    waypoints = np.column_stack((x_values, y_values))
    return waypoints


###############################
# Multi-DOF robot in 3D space #
###############################
def generate_line_between_two_points_3d(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64], points_count: int
) -> npt.NDArray[np.float64]:
    t_array = np.linspace(0, 1, points_count)
    return np.array([(1 - t) * p1 + t * p2 for t in t_array])  # parametric equation of a line


def generate_circle_3d(radius: float, points_count: int) -> npt.NDArray[np.float64]:
    t_array = np.linspace(0, 2 * np.pi, points_count)
    return np.array([[radius * np.cos(t), radius * np.sin(t), 0] for t in t_array])


def get_arm_states_from_cartesian_waypoints_3d(
    robot: Chain, cartesian_waypoints: list[list[float]]
) -> list[RobotState]:
    states: list[RobotState] = []
    joints_seed = None
    for x, y, z in cartesian_waypoints:
        transformation = np.eye(4, 4)
        transformation[:-1, 3:] = np.array([[x], [y], [z]])

        joints_angles: npt.NDArray[np.float64] = robot.inverse_kinematics_frame(
            target=transformation, initial_position=joints_seed
        )
        joints_seed = joints_angles
        theta1, theta2, theta3 = joints_angles[1:-1]
        states.append(RobotState([theta1, theta2, theta3]))
    return states


def forward_kinematics_3d(kinematic_chain: Chain, joint_angles: list[float]) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    frame_matrix = np.eye(4)
    for index, (link, joint_parameters) in enumerate(zip(kinematic_chain.links, joint_angles)):
        frame_matrix = np.dot(frame_matrix, np.asarray(link.get_link_frame_matrix(joint_parameters)))
        positions.append(frame_matrix[:3, 3])
    return positions[1:]
