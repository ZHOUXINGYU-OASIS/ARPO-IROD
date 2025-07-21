import daceypy_import_helper  # noqa: F401
from typing import Callable, Type
import numpy as np
import math
from daceypy import DA, RK, array, integrator

miuE = 398600.435436096  # gravitational constant of the Earth

def TBP_dynamics(
        t: float,
        x: np.array,
        miu: float = miuE,
) -> np.array:
    """
    Two-body dynamics (without time scaling)
    :param t: time epoch [t0, tf]
    :param x: orbital state
    :return: dx: state derivatives
    """
    pos: np.array = x[:3]
    vel: np.array = x[3:]
    r = np.linalg.norm(pos)
    acc: np.array = -miu * pos / (r ** 3)
    dx = np.concatenate((vel, acc))
    return dx


def TBP(
        x: array,
        t: float,
        miu: float = miuE,
) -> array:
    """
    Two-body dynamics without time scaling (under the framework of DA)
    :param x: orbital state
    :param t: time epoch [t0, tf]
    :return: dx: state derivatives
    """
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -miu * pos / (r ** 3)
    dx = vel.concat(acc)
    return dx

def TBP_time(
        x: array,
        tau: float,
        t0: DA,
        tf: DA,
        miu: float = miuE,
) -> array:
    """
    Two-body dynamics with time scaling (under the framework of DA)
    :param x: orbital state
    :param tau: time epoch [0, 1]
    :param t0: initial epoch
    :param tf: final epoch
    :return: dx: state derivatives
    """
    # input time tau is normalized. To retrieve t: tau * (tf - t0)
    # RHS of ODE must be multiplied by (tf - t0) to scale
    # t is computed but useless in case of autonomous dynamics
    t = tau * (tf - t0)  # epoch t is not employed in the two-body dynamics
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -miu * pos / (r ** 3)
    dx = (tf - t0) * (vel.concat(acc))
    return dx