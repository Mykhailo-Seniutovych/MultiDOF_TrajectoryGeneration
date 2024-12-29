import numpy as np
import numpy.typing as npt
from motion_profiles import JointMotionProfile


#################################
# MAX ALLOWED START SPEED #######
#################################
def calculate_max_allowed_start_speed(dist: float, v_max: float, a_max: float) -> float:
    """Calculate the maximum allowed start speed so if we always have enough time to decelerate to 0 and not travel more than the distance we have to travel."""

    acc = -np.abs(a_max) if dist >= 0 else np.abs(a_max)  # we either decelerate or accelerate
    t = np.sqrt(2 * dist / -acc)
    v_start_max = -acc * t
    v_start_max = np.clip(v_start_max, -np.abs(v_max), np.abs(v_max))
    return v_start_max


#################################
# MINIMAL TIME CALCULATIONS #####
#################################
def _calculate_increase_const_decrease_min_time(
    dist: float, v_start: float, v_final: float, v_max: float, a_max: float
) -> float:
    a_max = np.abs(a_max) if dist >= 0 else -np.abs(a_max)
    v_max = np.abs(v_max) if dist >= 0 else -np.abs(v_max)
    t_acc = (v_max - v_start) / a_max
    t_dec = (v_max - v_final) / a_max
    s_acc = v_start * t_acc + 0.5 * a_max * t_acc**2
    s_dec = v_max * t_dec - 0.5 * a_max * t_dec**2
    s_const = dist - s_acc - s_dec

    # distances must have same signs
    if not ((s_acc >= 0 and s_const >= 0 and s_dec >= 0) or (s_acc <= 0 and s_const <= 0 and s_dec <= 0)):
        return -1
    t_const = s_const / v_max
    T_min = t_acc + t_const + t_dec
    return T_min


def _calculate_increase_decrease_min_time(
    dist: float, v_start: float, v_final: float, v_max: float, a_max: float
) -> float:
    a_max = np.abs(a_max) if dist >= 0 else -np.abs(a_max)
    v_max = np.abs(v_max) if dist >= 0 else -np.abs(v_max)
    v_peak = np.sqrt(a_max * dist + (v_start**2 + v_final**2) / 2)
    v_peak = v_peak if dist >= 0 else -v_peak
    if np.abs(v_peak) > np.abs(v_max):
        return -1

    T_min = (2 * v_peak - v_start - v_final) / a_max
    return T_min


def _calculate_increase_min_time(dist: float, v_start: float, v_final: float, v_max: float, a_max: float) -> float:
    a_max = np.abs(a_max) if dist >= 0 else -np.abs(a_max)
    v_max = np.abs(v_max) if dist >= 0 else -np.abs(v_max)
    T_min = (-v_start + np.sqrt(v_start**2 + 2 * a_max * dist)) / a_max
    v_f_calculated = v_start + a_max * T_min
    if np.abs(v_f_calculated) > np.abs(v_final):
        return -1
    return T_min


def calculate_min_time(dist: float, v_start: float, v_final: float, v_max: float, a_max: float) -> float:
    assert v_start * v_final >= 0  # same sign
    T_min = _calculate_increase_const_decrease_min_time(dist, v_start, v_final, v_max, a_max)
    if T_min >= 0:
        T_min += 0.0000000001  # to avoid numerical errors
        return T_min

    T_min = _calculate_increase_decrease_min_time(dist, v_start, v_final, v_max, a_max)
    if T_min >= 0:
        T_min += 0.0000000001
        return T_min

    T_min = _calculate_increase_min_time(dist, v_start, v_final, v_max, a_max)
    if T_min >= 0:
        T_min += 0.0000000001
        return T_min

    assert False, "No solution found"


#################################
# MOTION PROFILES CALCULATIONS ##
#################################
def _calculate_increase_const_decrease_mp(
    v_start: float, v_final: float, dist: float, v_max: float, a_max: float, T: float
) -> float:
    # we need to know whether we will be accelerating then decelerating or vice versa
    # we will calculate how much distance we will travel if we just accelerated from V_0 to V_f
    # if the distance is less than S, we will be accelerating then decelerating, otherwise vice versa
    s_with_const_acc = v_start * T + 0.5 * (v_final - v_start) * T
    a_max = np.abs(a_max) if dist >= s_with_const_acc else -np.abs(a_max)

    def calc_motion_profile(v_const: float) -> JointMotionProfile:
        t_acc = np.abs((v_const - v_start) / a_max)
        t_dec = np.abs((v_final - v_const) / -a_max)
        t_const = T - t_acc - t_dec
        if t_const < 0:
            return None
        # if acceleration does not match the direction of velocity change
        if v_const < v_start and a_max > 0 or v_const > v_start and a_max < 0:
            return None

        # in such movement velocity cannot be between v_start and v_final
        min_v = np.min([np.abs(v_start), np.abs(v_final)])
        max_v = np.max([np.abs(v_start), np.abs(v_final)])
        if np.abs(v_const) < max_v and np.abs(v_const) > min_v:
            return None

        s_acc = v_start * t_acc + 0.5 * a_max * t_acc**2
        s_const = v_const * t_const
        s_dec = v_const * t_dec + 0.5 * -a_max * t_dec**2
        # distances must have same signs
        if not ((s_acc >= 0 and s_const >= 0 and s_dec >= 0) or (s_acc <= 0 and s_const <= 0 and s_dec <= 0)):
            return None

        return JointMotionProfile(
            v_start=v_start,
            v_const=v_const,
            v_final=v_final,
            acc1=a_max,
            acc2=-a_max,
            t_acc1=t_acc,
            t_const=t_const,
            t_acc2=t_dec,
            s_acc1=s_acc,
            s_const=s_const,
            s_acc2=s_dec,
        )

    D = (a_max * T + v_start + v_final) ** 2 - 2 * (v_start**2 + v_final**2 + 2 * a_max * dist)
    if D <= 0 or D == np.nan:
        return None

    v_const_1 = (a_max * T + v_start + v_final + np.sqrt(D)) / 2
    v_const_2 = (a_max * T + v_start + v_final - np.sqrt(D)) / 2
    profile1 = calc_motion_profile(v_const_1)
    profile2 = calc_motion_profile(v_const_2)
    profile = profile1 if profile1 is not None else profile2
    if profile is None:
        return None

    if abs(profile.v_const) > abs(v_max):
        return None
    return profile


def _calculate_increase_const_mp(
    v_start: float,
    v_final: float,
    dist: float,
    v_max: float,
    a_max: float,
    T: float,
) -> JointMotionProfile:
    is_inverse = abs(v_final) < abs(v_start)
    if is_inverse:
        v_start, v_final = v_final, v_start

    def calc_motion_profile(acc: float) -> JointMotionProfile:
        t_acc = np.abs((v_final - v_start) / acc)
        t_const = T - t_acc
        if t_const < 0:
            return None
        s_acc = v_start * t_acc + 0.5 * acc * t_acc**2
        s_const = v_final * t_const

        # if acceleration does not match the direction of velocity change
        if v_final < v_start and acc > 0 or v_final > v_start and acc < 0:
            return None

        if not is_inverse:
            return JointMotionProfile(
                v_start=v_start,
                v_const=v_final,
                v_final=v_final,
                acc1=acc,
                acc2=0,
                t_acc1=t_acc,
                t_const=t_const,
                t_acc2=0,
                s_acc1=s_acc,
                s_const=s_const,
                s_acc2=0,
            )
        else:
            return JointMotionProfile(
                v_start=v_final,
                v_const=v_final,
                v_final=v_start,
                acc1=0,
                acc2=-acc,
                t_acc1=0,
                t_const=t_const,
                t_acc2=t_acc,
                s_acc1=0,
                s_const=s_const,
                s_acc2=s_acc,
            )

    a = 0.5 * (v_final - v_start) ** 2 / (v_final * T - dist)
    if np.abs(a) > np.abs(a_max):
        return None

    if a == 0:
        return None

    if np.isnan(a):
        return None

    profile = calc_motion_profile(a)
    if profile is None:
        return None

    if abs(profile.v_const) > abs(v_max):
        return None
    return profile


def _calculate_increase_mp(
    v_start: float,
    v_final_max: float,
    dist: float,
    v_max: float,
    a_max: float,
    T: float,
) -> JointMotionProfile:
    a = 2 * (dist - v_start * T) / T**2
    if np.abs(a) > np.abs(a_max):
        return None

    v_final = v_start + a * T

    if np.abs(v_final) > np.abs(v_max):
        return None

    if np.abs(v_final) > np.abs(v_final_max):
        return None

    if v_final * v_start < 0:
        return None

    return JointMotionProfile(
        v_start=v_start,
        v_const=v_final,
        v_final=v_final,
        acc1=a,
        acc2=0,
        t_acc1=T,
        t_const=0,
        t_acc2=0,
        s_acc1=dist,
        s_const=0,
        s_acc2=0,
    )


def _calculate_rapid_decrease_mp(v_start: float, dist: float, a_max: float, T: float) -> JointMotionProfile:
    a = v_start**2 / (-2 * dist)
    allowed_float_calc_err = 0.0000000000001  # because of float point errors
    if np.abs(a) - np.abs(a_max) > allowed_float_calc_err:
        return None
    t = np.sqrt(2 * dist / (-a))
    if t >= T:
        return None

    v_final = v_start + a * t
    v_final = v_final if np.abs(v_final) > allowed_float_calc_err else 0
    if v_start * v_final < 0:
        return None

    if np.abs(v_final) > 1e-9:
        return None

    return JointMotionProfile(
        v_start=v_start,
        v_const=0,
        v_final=0,
        acc1=a,
        acc2=0,
        t_acc1=t,
        t_const=T - t,
        t_acc2=0,
        s_acc1=dist,
        s_const=0,
        s_acc2=0,
    )


def calculate_mp(
    v_start: float, v_final: float, dist: float, v_max: float, a_max: float, T: float
) -> JointMotionProfile:
    assert v_start * v_final >= 0
    assert T > 0
    profile = _calculate_increase_const_mp(v_start=v_start, v_final=v_final, dist=dist, v_max=v_max, a_max=a_max, T=T)
    if profile is not None:
        return profile

    profile = _calculate_increase_const_decrease_mp(
        v_start=v_start, v_final=v_final, dist=dist, a_max=a_max, v_max=v_max, T=T
    )
    if profile is not None:
        return profile

    profile = _calculate_increase_mp(v_start=v_start, v_final_max=v_final, dist=dist, v_max=v_max, a_max=a_max, T=T)
    if profile is not None:
        return profile

    profile = _calculate_rapid_decrease_mp(v_start=v_start, dist=dist, a_max=a_max, T=T)
    assert profile is not None
    return profile
