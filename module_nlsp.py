"""Nonlinear Least-Squares Problem"""
import numpy as np
import random
import math
import warnings
import time
import scipy
import scipy.io as scio
from scipy.integrate import solve_ivp
import daceypy_import_helper  # noqa: F401
from typing import Callable
from daceypy import DA, array, ADS
from module_integrator import RK78
from module_measurements import generate_polynomials, generate_los_polynomials, get_stm_stt_coefficients, skew_symmetric_matrix

warnings.filterwarnings("ignore")
RelTol = 1e-12
AbsTol = 1e-12
default_std = 1e-4

def optimal_linear_orbit_determination(
        initial_guess: np.array,
        t_series: np.array,
        chief_states: np.array,
        deputy_states: np.array,
        los_vectors: np.array,
        f: Callable[[float, np.array], np.array],
        ft: Callable[[array, float], array],
        order: int,
        max_iteration: int = 100,
        eps: float = 1e-8,
) -> tuple[np.array, np.array, np.array, int, np.array, np.array, float, int]:
    """Optimal linear orbit determination"""
    """
    [1] Sinclair, A. J., and Alan Lovell, T. “Optimal Linear Orbit Determination.” 
    Journal of Guidance, Control, and Dynamics, Vol. 43, No. 3, 2020, pp. 628–632. 
    https://doi.org/10.2514/1.G004182.
    """
    DIM = 6
    start = time.time()
    guess = initial_guess.copy()
    """Begin iteration"""
    tol = np.inf
    iteration = 0
    flag = 1
    solutions_history = np.zeros((max_iteration + 1, DIM))
    solutions_history[0] = guess
    try:
        if np.linalg.norm(guess - deputy_states[0]) >= 1.0:
            print("Too large RIOD errors.")
            raise ValueError("Too large RIOD errors.")
        while tol > eps:
            """Generate polynomials"""
            polynomials = generate_polynomials(
                initial_state=guess,
                t_series=t_series,
                ft=ft,
                order=order,
                if_push_order=True,
                reduced_order=1,
                if_preserve_constant=True,
            )
            """Generate STM"""
            STMs = np.zeros((len(t_series), DIM, DIM))
            estimated_states = np.zeros((len(t_series), DIM))
            for k in range(len(t_series)):
                STM, _ = get_stm_stt_coefficients(
                    polynomial=polynomials[k],
                    DIM=DIM,
                    order=1,
                )
                STMs[k] = STM
                estimated_states[k] = polynomials[k].cons()
            """Build matrix M and b"""
            M = np.zeros((DIM, DIM))
            b = np.zeros((DIM, 1))
            for k in range(len(t_series)):
                rho = estimated_states[k, :3] - chief_states[k, :3]
                los_cross = skew_symmetric_matrix(los_vectors[k])
                STM = STMs[k]
                STM_r = STM[:3, :]
                M -= STM_r.T @ los_cross @ los_cross @ STM_r
                b += STM_r.T @ los_cross @ los_cross @ rho.reshape(-1, 1)
            """Solve for the solution"""
            sol = (np.linalg.inv(M) @ b).reshape(DIM)
            guess += sol
            iteration += 1
            solutions_history[iteration] = guess
            tol = np.linalg.norm(sol)
            if iteration > (max_iteration - 1):
                # maximal iterations reached
                flag = 0
                break
            if (np.linalg.norm(guess - deputy_states[0]) >= np.linalg.norm(chief_states[0] - deputy_states[0])) and (iteration >= 5):
                # too large estimation errors
                flag = 0
                break
            if (np.linalg.norm(guess - deputy_states[0]) >= 10.0 * np.linalg.norm(chief_states[0] - deputy_states[0])) and (len(t_series) <= 5):
                # too large estimation errors
                flag = 0
                break
            if np.linalg.norm(guess - chief_states[0]) <= 1e-4:
                # converge to an undesired solution
                flag = 0
                break
    except:
        flag = 0
        iteration = 100
    time_cost = time.time() - start
    """Return results"""
    try:
        t0 = t_series[0]
        tf = t_series[-1]
        deputy = solve_ivp(f, [t0, tf], guess, args=(), method='RK45',
                           t_eval=t_series, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        estimated_states = deputy.y.T
        AE = abs(estimated_states - deputy_states)
        RE = np.where(deputy_states == 0, np.nan,
                      np.abs(estimated_states - deputy_states) / np.abs(deputy_states) * 100)
    except:
        estimated_states = np.zeros((len(t_series), DIM))
        AE = abs(estimated_states - deputy_states)
        RE = np.where(deputy_states == 0, np.nan,
                      np.abs(estimated_states - deputy_states) / np.abs(deputy_states) * 100)
    return guess, estimated_states, solutions_history, iteration, AE, RE, time_cost, flag

def nonlinear_least_squares_orbit_determination(
        initial_guess: np.array,
        t_series: np.array,
        chief_states: np.array,
        deputy_states: np.array,
        los_vectors: np.array,
        f: Callable[[float, np.array], np.array],
        ft: Callable[[array, float], array],
        order: int,
        max_iteration: int = 100,
        eps: float = 1e-8,
        timeout: float = 30.0,
) -> tuple[np.array, np.array, np.array, int, np.array, np.array, float, int]:
    """Nonlinear least-squares (NLS)"""
    DIM = 6
    DIMz = 3
    start = time.time()
    guess = initial_guess.copy()
    """Begin iteration"""
    tol = np.inf
    iteration = 0
    flag = 1
    solutions_history = np.zeros((max_iteration + 1, DIM))
    solutions_history[0] = guess
    try:
        if np.linalg.norm(guess - deputy_states[0]) >= 1.0:
            print("Too large RIOD errors.")
            raise ValueError("Too large RIOD errors.")
        while tol > eps:
            """Generate polynomials"""
            polynomials = generate_polynomials(
                initial_state=guess,
                t_series=t_series,
                ft=ft,
                order=order,
                if_push_order=True,
                reduced_order=1,
                if_preserve_constant=True,
            )
            """Generate STM"""
            los_polynomials = generate_los_polynomials(
                chief_states=chief_states,
                t_series=t_series,
                polynomials=polynomials,
            )
            STMs = np.zeros((len(t_series), DIMz, DIM))
            estimated_states = np.zeros((len(t_series), DIMz))
            for k in range(len(t_series)):
                for k1 in range(DIMz):
                    for k2 in range(DIM):
                        STMs[k, k1, k2] = los_polynomials[k][k1].deriv(k2 + 1).cons()
                relative_pos = (polynomials[k].cons() - chief_states[k])[:3]
                estimated_states[k] = relative_pos / np.linalg.norm(relative_pos)
            """Build matrix M and b"""
            M = np.zeros((len(t_series) * DIMz, DIM))
            b = np.zeros((len(t_series) * DIMz, 1))
            for k in range(len(t_series)):
                M[(k * DIMz):((k + 1) * DIMz), :] = STMs[k]
                b[(k * DIMz):((k + 1) * DIMz), :] = (los_vectors[k] - estimated_states[k]).reshape(DIMz, 1)
            """Solve for the solution"""
            sol = (np.linalg.inv(M.T @ M) @ M.T @ b).reshape(DIM)
            guess += sol
            iteration += 1
            solutions_history[iteration] = guess
            tol = np.linalg.norm(sol)
            if iteration > (max_iteration - 1):
                # maximal iterations reached
                flag = 0
                break
            if (np.linalg.norm(guess - deputy_states[0]) >= np.linalg.norm(chief_states[0] - deputy_states[0])) and (iteration >= 5):
                # too large estimation errors
                flag = 0
                break
            if (np.linalg.norm(guess - deputy_states[0]) >= 10.0 * np.linalg.norm(chief_states[0] - deputy_states[0])) and (len(t_series) <= 5):
                # too large estimation errors
                flag = 0
                break
            if np.linalg.norm(guess - chief_states[0]) <= 1e-4:
                # converge to an undesired solution
                flag = 0
                break
            """Should not exceed 30 s"""
            if time.time() - start >= timeout:
                print("Maximal CPU time reached.")
                raise ValueError("Maximal CPU time reached.")
    except:
        flag = 0
        iteration = 100
    time_cost = time.time() - start
    """Return results"""
    try:
        t0 = t_series[0]
        tf = t_series[-1]
        deputy = solve_ivp(f, [t0, tf], guess, args=(), method='RK45',
                           t_eval=t_series, max_step=np.inf, rtol=RelTol, atol=AbsTol)
        estimated_states = deputy.y.T
        AE = abs(estimated_states - deputy_states)
        RE = np.where(deputy_states == 0, np.nan,
                      np.abs(estimated_states - deputy_states) / np.abs(deputy_states) * 100)
    except:
        estimated_states = np.zeros((len(t_series), DIM))
        AE = abs(estimated_states - deputy_states)
        RE = np.where(deputy_states == 0, np.nan,
                      np.abs(estimated_states - deputy_states) / np.abs(deputy_states) * 100)
    return guess, estimated_states, solutions_history, iteration, AE, RE, time_cost, flag



