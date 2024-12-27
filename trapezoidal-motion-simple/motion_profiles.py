import numpy as np
import numpy.typing as npt


_ALLOWED_FLOAT_CALC_ERR = 0.0000000000001


class JointMotionProfile:
    def __init__(self, v_peak: float, acc: float, t_acc: float, t_peak: float, s_acc: float, s_peak: float):
        assert t_acc >= 0
        assert t_peak >= 0
        assert np.isclose(s_acc, acc * t_acc**2 / 2, atol=_ALLOWED_FLOAT_CALC_ERR)
        assert np.isclose(s_peak, v_peak * t_peak, atol=_ALLOWED_FLOAT_CALC_ERR)

        self.v_peak = v_peak
        self.acc = acc
        self.t_acc = t_acc
        self.t_peak = t_peak
        self.s_acc = s_acc
        self.s_peak = s_peak

    def total_distance(self) -> float:
        return self.s_acc * 2 + self.s_peak

    def total_time(self) -> float:
        return self.t_acc * 2 + self.t_peak


class RobotMotionProfile:
    def __init__(
        self,
        joint_mps: list[JointMotionProfile],
        initial_time: float,
        initial_pos: npt.NDArray[np.float64],
    ):
        self.joint_mps = joint_mps
        self.initial_time = initial_time
        self.initial_pos = np.copy(initial_pos)
        self.profile_time = np.max([joint_mp.total_time() for joint_mp in self.joint_mps])

    def total_time(self) -> float:
        return self.profile_time + self.initial_time

    def total_distances(self) -> npt.NDArray[np.float64]:
        return self.initial_pos + np.array([joint_mp.total_distance() for joint_mp in self.joint_mps])

    def position_at(self, time: float) -> npt.NDArray[np.float64]:
        joint_positions = np.zeros(len(self.joint_mps))
        dt = time - self.initial_time
        assert dt < self.profile_time + _ALLOWED_FLOAT_CALC_ERR
        for joint_index, joint_mp in enumerate(self.joint_mps):
            assert dt >= -_ALLOWED_FLOAT_CALC_ERR
            if dt <= joint_mp.t_acc:
                joint_positions[joint_index] = self.initial_pos[joint_index] + 0.5 * joint_mp.acc * dt**2
            elif dt <= joint_mp.t_acc + joint_mp.t_peak:
                joint_positions[joint_index] = (
                    self.initial_pos[joint_index] + joint_mp.s_acc + joint_mp.v_peak * (dt - joint_mp.t_acc)
                )
            elif dt <= joint_mp.total_time():
                joint_positions[joint_index] = (
                    self.initial_pos[joint_index]
                    + joint_mp.s_acc
                    + joint_mp.s_peak
                    + joint_mp.v_peak * (dt - joint_mp.t_acc - joint_mp.t_peak)
                    - 0.5 * joint_mp.acc * (dt - joint_mp.t_acc - joint_mp.t_peak) ** 2
                )
            else:
                joint_positions[joint_index] = (
                    self.initial_pos[joint_index] + joint_mp.s_acc + joint_mp.s_peak + joint_mp.s_acc
                )
        return joint_positions

    def velocity_at(self, time: float) -> npt.NDArray[np.float64]:
        joint_velocities = np.zeros(len(self.joint_mps))
        dt = time - self.initial_time
        assert dt < self.profile_time + _ALLOWED_FLOAT_CALC_ERR
        for joint_index, joint_mp in enumerate(self.joint_mps):
            assert dt >= -_ALLOWED_FLOAT_CALC_ERR
            if dt <= joint_mp.t_acc:
                joint_velocities[joint_index] = joint_mp.acc * dt
            elif dt <= joint_mp.t_acc + joint_mp.t_peak:
                joint_velocities[joint_index] = joint_mp.v_peak
            elif dt <= joint_mp.total_time():
                joint_velocities[joint_index] = joint_mp.v_peak - joint_mp.acc * (dt - joint_mp.t_acc - joint_mp.t_peak)
            else:
                joint_velocities[joint_index] = 0
        return joint_velocities

    def acceleration_at(self, time: float) -> npt.NDArray[np.float64]:
        joint_accelerations = np.zeros(len(self.joint_mps))
        dt = time - self.initial_time
        assert dt < self.profile_time + _ALLOWED_FLOAT_CALC_ERR
        for joint_index, joint_mp in enumerate(self.joint_mps):
            assert dt >= -_ALLOWED_FLOAT_CALC_ERR
            if dt <= joint_mp.t_acc:
                joint_accelerations[joint_index] = joint_mp.acc
            elif dt <= joint_mp.t_acc + joint_mp.t_peak:
                joint_accelerations[joint_index] = 0
            elif dt <= joint_mp.total_time():
                joint_accelerations[joint_index] = -joint_mp.acc
            else:
                joint_accelerations[joint_index] = 0
        return joint_accelerations
