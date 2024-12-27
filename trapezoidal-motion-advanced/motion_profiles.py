import numpy as np
import numpy.typing as npt


_ALLOWED_FLOAT_CALC_ERR = 0.0000000000001


class JointMotionProfile:
    def __init__(
        self,
        v_start: float,
        v_const: float,
        v_final: float,
        acc1: float,
        acc2: float,
        t_acc1: float,
        t_const: float,
        t_acc2: float,
        s_acc1: float,
        s_const: float,
        s_acc2: float,
    ):
        assert t_acc1 >= 0
        assert t_const >= 0
        assert t_acc2 >= 0

        assert np.isclose(s_acc1, v_start * t_acc1 + acc1 * t_acc1**2 / 2, atol=_ALLOWED_FLOAT_CALC_ERR)
        assert np.isclose(s_const, v_const * t_const, atol=_ALLOWED_FLOAT_CALC_ERR)
        assert np.isclose(s_acc2, v_const * t_acc2 + acc2 * t_acc2**2 / 2, atol=_ALLOWED_FLOAT_CALC_ERR)
        assert np.isclose(v_final, v_const + acc2 * t_acc2, atol=_ALLOWED_FLOAT_CALC_ERR)

        self.v_start = v_start
        self.v_const = v_const
        self.v_final = v_final

        self.acc1 = acc1
        self.acc2 = acc2

        self.t_acc1 = t_acc1
        self.t_const = t_const
        self.t_acc2 = t_acc2

        self.s_acc1 = s_acc1
        self.s_const = s_const
        self.s_acc2 = s_acc2

    def total_distance(self) -> float:
        return self.s_acc1 + self.s_const + self.s_acc2

    def total_time(self) -> float:
        return self.t_acc1 + self.t_const + self.t_acc2


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
        for index in range(len(self.joint_mps) - 1):
            assert np.isclose(
                self.joint_mps[index].total_time(), self.joint_mps[index + 1].total_time(), atol=_ALLOWED_FLOAT_CALC_ERR
            )
        self.profile_time = self.joint_mps[0].total_time()

    def total_time(self) -> float:
        return self.profile_time + self.initial_time

    def total_distances(self) -> npt.NDArray[np.float64]:
        return self.initial_pos + np.array([joint_mp.total_distance() for joint_mp in self.joint_mps])

    def position_at(self, time: float) -> npt.NDArray[np.float64]:
        joint_positions = np.zeros(len(self.joint_mps))
        dt = time - self.initial_time
        assert dt < self.profile_time + _ALLOWED_FLOAT_CALC_ERR
        assert dt >= -_ALLOWED_FLOAT_CALC_ERR
        for joint_index, joint_mp in enumerate(self.joint_mps):
            if dt <= joint_mp.t_acc1:
                joint_positions[joint_index] = (
                    self.initial_pos[joint_index] + joint_mp.v_start * dt + 0.5 * joint_mp.acc1 * dt**2
                )
            elif dt <= joint_mp.t_acc1 + joint_mp.t_const:
                joint_positions[joint_index] = (
                    self.initial_pos[joint_index] + joint_mp.s_acc1 + joint_mp.v_const * (dt - joint_mp.t_acc1)
                )
            elif dt <= joint_mp.total_time() + _ALLOWED_FLOAT_CALC_ERR:
                joint_positions[joint_index] = (
                    self.initial_pos[joint_index]
                    + joint_mp.s_acc1
                    + joint_mp.s_const
                    + joint_mp.v_const * (dt - joint_mp.t_acc1 - joint_mp.t_const)
                    + 0.5 * joint_mp.acc2 * (dt - joint_mp.t_acc1 - joint_mp.t_const) ** 2
                )
            else:
                assert False, "time is out of bounds"
        return joint_positions

    def velocity_at(self, time: float) -> npt.NDArray[np.float64]:
        joint_velocities = np.zeros(len(self.joint_mps))
        dt = time - self.initial_time
        assert dt < self.profile_time + _ALLOWED_FLOAT_CALC_ERR
        for joint_index, joint_mp in enumerate(self.joint_mps):
            assert dt >= -_ALLOWED_FLOAT_CALC_ERR
            if dt <= joint_mp.t_acc1:
                joint_velocities[joint_index] = joint_mp.v_start + joint_mp.acc1 * dt
            elif dt <= joint_mp.t_acc1 + joint_mp.t_const:
                joint_velocities[joint_index] = joint_mp.v_const
            elif dt <= joint_mp.total_time() + _ALLOWED_FLOAT_CALC_ERR:
                joint_velocities[joint_index] = joint_mp.v_const + joint_mp.acc2 * (
                    dt - joint_mp.t_acc1 - joint_mp.t_const
                )
            else:
                assert False, "time is out of bounds"
        return joint_velocities

    def acceleration_at(self, time: float) -> npt.NDArray[np.float64]:
        joint_accelerations = np.zeros(len(self.joint_mps))
        dt = time - self.initial_time
        assert dt < self.profile_time + _ALLOWED_FLOAT_CALC_ERR
        for joint_index, joint_mp in enumerate(self.joint_mps):
            assert dt >= -_ALLOWED_FLOAT_CALC_ERR
            if dt <= joint_mp.t_acc1:
                joint_accelerations[joint_index] = joint_mp.acc1
            elif dt <= joint_mp.t_acc1 + joint_mp.t_const:
                joint_accelerations[joint_index] = 0
            elif dt <= joint_mp.total_time() + _ALLOWED_FLOAT_CALC_ERR:
                joint_accelerations[joint_index] = joint_mp.acc2
            else:
                assert False, "time is out of bounds"
        return joint_accelerations
