import bisect
from typing import List, Protocol
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from motion_profiles import RobotMotionProfile, JointMotionProfile
import matplotlib.pyplot as plt

from robot_state import RobotState, RobotStateDiff


class MotionProfileBuilder:
    def __init__(self, way_points: List[RobotState], max_vel: float, max_acc: float):
        assert len(way_points) > 1
        self.__max_vel = np.abs(max_vel)
        self.__max_acc = np.abs(max_acc)
        self.__way_points = way_points

    def build_mp_map(self) -> OrderedDict[float, RobotMotionProfile]:
        time = 0
        distances = np.array([0.0] * self.__way_points[0].dof())
        multi_dof_mp_map: OrderedDict[float, RobotMotionProfile] = OrderedDict({})
        for index in range(len(self.__way_points) - 1):
            state_diff = self.__way_points[index + 1] - self.__way_points[index]
            robot_mp = self.__calculate_motion_profile(state_diff, time, distances)
            multi_dof_mp_map[time] = robot_mp
            time = robot_mp.total_time()
            distances = robot_mp.total_distances()
        return multi_dof_mp_map

    def __calculate_motion_profile(
        self, state_diff: RobotStateDiff, initial_time: float, initial_dist: npt.NDArray[np.float64]
    ) -> RobotMotionProfile:
        # this list will represent motion profiles for each joint of the robot
        joint_mps: list[JointMotionProfile] = []
        for single_joint_dist in state_diff.values():
            max_vel = self.__max_vel if single_joint_dist >= 0 else -self.__max_vel
            max_acc = self.__max_acc if single_joint_dist >= 0 else -self.__max_acc

            time_acc = max_vel / max_acc
            distance_acc = 0.5 * max_acc * time_acc**2
            # during deceleration we will accelerate with the same acceleration,
            # just the opposite sign, so we will travel the same distance
            distance_dec = distance_acc

            # if the distance we travel to reach max velocity and then go back to 0
            # is greater than the total distance we need to travel,
            # we will use the triangular motion profile instead of the trapezoidal motion profile
            # meaning we will reach some peak velocity less than max velocity and then go back to 0
            is_triangular_movement = np.abs(distance_acc + distance_dec) > np.abs(single_joint_dist)
            if is_triangular_movement:
                distance_acc = single_joint_dist / 2
                time_acc = np.sqrt(2 * distance_acc / max_acc)
                vel_peak = max_acc * time_acc
                joint_mps.append(
                    JointMotionProfile(
                        v_peak=vel_peak, acc=max_acc, t_acc=time_acc, t_peak=0, s_acc=distance_acc, s_peak=0
                    )
                )
            else:
                distance_peak = single_joint_dist - distance_acc - distance_dec
                time_peak = distance_peak / max_vel
                joint_mps.append(
                    JointMotionProfile(
                        v_peak=max_vel,
                        acc=max_acc,
                        t_acc=time_acc,
                        t_peak=time_peak,
                        s_acc=distance_acc,
                        s_peak=distance_peak,
                    )
                )

        return RobotMotionProfile(joint_mps, initial_time, initial_dist)


class Trajectory:
    def __init__(self, way_points: list[RobotState], max_vel: float, max_acc: float):
        assert len(way_points) > 2
        # each time point has a list of motion profiles for each dimension (arm joint in our case)
        self.__multidof_mp_map: OrderedDict[float, RobotMotionProfile] = MotionProfileBuilder(
            way_points, max_vel, max_acc
        ).build_mp_map()
        self.__initial_diff = way_points[0] - RobotState(np.zeros(way_points[0].dof()))

    def position_at(self, time: float) -> RobotState:
        keys = list(self.__multidof_mp_map.keys())
        left_time_index = bisect.bisect(keys, time) - 1
        left_time = keys[left_time_index]
        assert left_time <= time
        robot_mp = self.__multidof_mp_map[left_time]
        joint_values = robot_mp.position_at(time)
        return RobotState(joint_values) + self.__initial_diff

    # traces trajectory with a given step time interval, meaning it will return a list of robot states for each time from the trajectory
    def trace(
        self, time_step: float, start_time: float = None, end_time: float = None
    ) -> list[tuple[float, RobotState]]:
        start_time = start_time or 0
        end_time = end_time or self.total_time()
        assert start_time >= 0 and end_time >= 0 and start_time < end_time and end_time <= self.total_time()
        assert time_step > 0 and time_step < end_time - start_time

        states = [(time, self.position_at(time)) for time in np.arange(start_time, end_time, time_step)]
        states.append((end_time, self.position_at(end_time)))
        return states

    def get_motion_profiles(self) -> OrderedDict[float, RobotMotionProfile]:
        return self.__multidof_mp_map

    def total_time(self) -> float:
        last_time_key = list(self.__multidof_mp_map.keys())[-1]
        last_profile = self.__multidof_mp_map[last_time_key]
        return last_profile.total_time()
