import bisect
from typing import List, Protocol
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from motion_profiles import RobotMotionProfile
import matplotlib.pyplot as plt

from robot_state import RobotState
from motion_profiles_calc import calculate_max_allowed_start_speed, calculate_min_time, calculate_mp


class MotionProfileBuilder:
    def __init__(self, way_points: List[RobotState], max_vel: float, max_acc: float):
        assert len(way_points) > 1
        self.__max_vel = np.abs(max_vel)
        self.__max_acc = np.abs(max_acc)
        self.__way_points = way_points
        self.__dof = way_points[0].dof()

    def build_mp_map(self) -> OrderedDict[float, RobotMotionProfile]:
        multi_dof_mp_map: OrderedDict[float, RobotMotionProfile] = OrderedDict({})

        max_start_speeds = self.__calculate_max_allowed_start_speeds()
        current_wp_start_speeds = max_start_speeds[0]
        prev_distances = np.zeros(self.__dof)
        current_time = 0
        for index in range(len(self.__way_points) - 1):
            diff = self.__way_points[index + 1] - self.__way_points[index]
            next_wp_max_start_speeds = max_start_speeds[index + 1]
            max_time_for_interval = np.max(
                [
                    calculate_min_time(
                        dist=diff.values()[dof],
                        v_start=current_wp_start_speeds[dof],
                        v_final=next_wp_max_start_speeds[dof],
                        v_max=self.__max_vel,
                        a_max=self.__max_acc,
                    )
                    for dof in range(self.__dof)
                ]
            )

            motion_profiles = [
                calculate_mp(
                    v_start=current_wp_start_speeds[dof],
                    v_final=next_wp_max_start_speeds[dof],
                    dist=diff.values()[dof],
                    v_max=self.__max_vel,
                    a_max=self.__max_acc,
                    T=max_time_for_interval,
                )
                for dof in range(self.__dof)
            ]

            current_wp_start_speeds = [mp.v_final for mp in motion_profiles]
            multi_dof_mp_map[current_time] = RobotMotionProfile(motion_profiles, current_time, prev_distances)

            current_time += max_time_for_interval
            prev_distances += np.array([mp.total_distance() for mp in motion_profiles])

        return multi_dof_mp_map

    def __calculate_max_allowed_start_speeds(self) -> list[npt.NDArray[np.float64]]:
        speeds = [np.zeros(self.__dof)]  # we should start from 0 speed
        prev_diff = self.__way_points[1] - self.__way_points[0]
        for index in range(1, len(self.__way_points) - 1):
            speeds.append(np.zeros(self.__dof))
            diff = self.__way_points[index + 1] - self.__way_points[index]
            for dof in range(self.__dof):
                is_direction_change = prev_diff.values()[dof] * diff.values()[dof] < 0
                if not is_direction_change:
                    max_start_speed = calculate_max_allowed_start_speed(
                        dist=diff.values()[dof], v_max=self.__max_vel, a_max=self.__max_acc
                    )
                    speeds[-1][dof] = max_start_speed
            prev_diff = diff
        speeds.append(np.zeros(self.__dof))  # we should stop at the last waypoint
        assert len(speeds) == len(self.__way_points)
        return speeds


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
