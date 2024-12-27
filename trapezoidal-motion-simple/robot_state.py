import numpy as np
import numpy.typing as npt


class RobotStateDiff:
    def __init__(self, values: np.ndarray[float, float]):
        self.__values = values

    def values(self) -> npt.NDArray[np.float64]:
        return self.__values

    def norm(self) -> float:
        return np.linalg.norm(self.values(), ord=2)
        # return np.max(np.abs(self.values()))

    def interpolate(self, alpha: float) -> "RobotStateDiff":
        assert 0 <= alpha <= 1
        return RobotStateDiff(self.values() * alpha)

    def __sub__(self, other: "RobotStateDiff") -> "RobotStateDiff":
        return RobotStateDiff(self.values() - other.values())

    def __str__(self):
        return f"joints: {self.__values}"


class RobotState:
    def __init__(self, joint_vals: npt.NDArray[np.float64]):
        self.__values = np.copy(joint_vals)

    def values(self) -> npt.NDArray[np.float64]:
        return self.__values

    def dof(self):
        return len(self.__values)

    def __add__(self, diff: RobotStateDiff) -> "RobotState":
        new_values: npt.NDArray[np.float64] = self.values() + diff.values()
        return RobotState(new_values)

    def __sub__(self, other: "RobotState") -> RobotStateDiff:
        return RobotStateDiff(self.values() - other.values())
