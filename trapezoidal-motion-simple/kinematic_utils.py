import numpy as np
import numpy.typing as npt

from robot_state import RobotState


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
