import numpy as np
from contextlib import contextmanager
import joblib
from tqdm import tqdm
import random
import math
import warnings
import time
import scipy
import scipy.io as scio
from scipy.integrate import solve_ivp
import daceypy_import_helper  # noqa: F401
import multiprocessing
from joblib import Parallel, delayed
from typing import Callable
from daceypy import DA, array, ADS
from module_irod import setup_seed, get_flyby_scenario, get_formation_scenario
from module_nlsp import optimal_linear_orbit_determination, nonlinear_least_squares_orbit_determination
from module_measurements import generate_polynomials, get_stm_stt_coefficients
from module_optimization import RIOD_convex_optimization
from module_constrained_optimization import RIOD_convex_constrained_optimization
from module_optimization_weight import RIOD_convex_optimization_weighted
from module_constrained_optimization_weight import RIOD_convex_constrained_optimization_weighted

warnings.filterwarnings("ignore")
RelTol = 1e-12
AbsTol = 1e-12
default_std = 1e-4

def example_run():
    """An example"""
    order = 5
    default_std = 1e-4
    weight_strategy: int = 0  # 0: original OLOD, 1: inverse-distance, 2: 2D tangent-plane
    xc0, xd0, t_series, los_vectors, f, ft, chief_states, deputy_states, R = get_flyby_scenario(
        num_observations=10,
        factor=10.0,
        revolution=1.0,
        std=default_std,
        if_add_noises=True,
        if_set_seed=True,
        seed=42,
    )
    dx0, solution, AE, RE, solutions_history, _, time_costs, minimal_distance, residuals = adaptive_recursive_optimization_method(
        xc0=xc0,
        xd0=xd0,
        R=R["LOS"],
        t_series=t_series,
        los_vectors=los_vectors,
        ft=ft,
        order=order,
        polynomials=None,
        max_iteration=100,
        eps=1e-8,
        constraints_bound=np.array([1e-3, 3e-3, 1e-1]),
        ifPrint=False,
        if_first_order_cost=True,
        if_add_weights=False,
        weight_strategy=weight_strategy,
    )
    initial_guess = xc0 + dx0
    """Implement ROD"""
    # refined_guess, estimated_states, _, iteration, _, _, time_cost, flag = optimal_linear_orbit_determination(
    #     initial_guess=initial_guess,
    #     t_series=t_series,
    #     chief_states=chief_states,
    #     deputy_states=deputy_states,
    #     los_vectors=los_vectors,
    #     f=f,
    #     ft=ft,
    #     order=order,
    # )
    refined_guess, estimated_states, _, iteration, _, _, time_cost, flag = nonlinear_least_squares_orbit_determination(
        initial_guess=initial_guess,
        t_series=t_series,
        chief_states=chief_states,
        deputy_states=deputy_states,
        los_vectors=los_vectors,
        f=f,
        ft=ft,
        order=order,
    )
    return refined_guess, estimated_states, iteration, time_cost, flag

def first_step_optimization(
        xc0: np.array,
        xd0: np.array,
        R: np.array,
        initial_guess: np.array,
        t_series: np.array,
        los_vectors: np.array,
        ft: Callable[[array, float], array],
        order: int,
        polynomials: list = None,
        minimal_distance: float = 1e-3,
        max_iteration: int = 100,
        eps: float = 1e-6,
        ifPrint: bool = True,
        if_first_order_cost: bool = True,
        if_add_weights: bool = False,
        weight_strategy: int = 0,  # 0: original OLOD, 1: inverse-distance, 2: 2D tangent-plane
        residual_order: float = 0.5,
) -> tuple[np.array, np.array, np.array, np.array, np.array, int, float]:
    """First-step optimization method for initial relative orbit determination"""
    DIM = 6
    start = time.time()
    """Generate polynomials"""
    if polynomials is None:
        polynomials = generate_polynomials(
            initial_state=xc0,
            t_series=t_series,
            ft=ft,
            order=order,
            if_push_order=False,
        )
    time_cost = time.time() - start
    reference_point = initial_guess.copy()
    """Begin recursive optimization"""
    tol = np.inf
    iteration = 0
    flag = 1
    solutions_history = np.zeros([max_iteration, DIM])
    Maps = polynomials.copy()
    while tol > eps:
        with DA.cache_manager():  # optional, for efficiency
            for k in range(len(polynomials)):
                Maps[k] = polynomials[k].eval(reference_point + array.identity(DIM))
        """Generate STMs and STTs"""
        start = time.time()
        STMs = np.zeros([len(t_series), DIM, DIM])
        for k in range(len(t_series)):
            STM, _ = get_stm_stt_coefficients(
                polynomial=Maps[k],
                DIM=DIM,
                order=1,
            )
            STMs[k] = STM
        """Implement recursive optimization algorithm"""
        if if_add_weights is False:
            dx0 = RIOD_convex_constrained_optimization(
                guess=reference_point,
                STMs=STMs,
                Maps=Maps,
                los_vectors=los_vectors,
                ifPrint=ifPrint,
                if_first_order_cost=if_first_order_cost,
                minimal_distance=minimal_distance,
                residual_order=residual_order,
            )
        else:
            dx0 = RIOD_convex_constrained_optimization_weighted(
                guess=reference_point,
                STMs=STMs,
                Maps=Maps,
                los_vectors=los_vectors,
                R=R,
                weight_strategy=weight_strategy,
                ifPrint=ifPrint,
                if_first_order_cost=if_first_order_cost,
                minimal_distance=minimal_distance,
            )
        time_cost += time.time() - start
        """Update solutions"""
        reference_point += dx0
        solutions_history[iteration] = reference_point
        iteration += 1
        tol = np.linalg.norm(dx0)
        if iteration >= max_iteration:
            flag = 0
            break
    """Generate solutions"""
    solution = xc0 + reference_point
    true_states = np.concatenate((xd0, (xd0 - xc0)))
    AE = abs(np.concatenate((solution, reference_point)) - true_states)
    RE = np.where(true_states == 0, np.nan, AE / np.abs(true_states) * 100)
    """Return results"""
    dx0 = reference_point
    return dx0, solution, AE, RE, solutions_history, flag, time_cost

def second_step_optimization(
        xc0: np.array,
        xd0: np.array,
        R: np.array,
        initial_guess: np.array,
        t_series: np.array,
        los_vectors: np.array,
        ft: Callable[[array, float], array],
        order: int,
        polynomials: list = None,
        max_iteration: int = 100,
        eps: float = 1e-6,
        ifPrint: bool = True,
        if_first_order_cost: bool = True,
        if_add_weights: bool = False,
        weight_strategy: int = 0,  # 0: original OLOD, 1: inverse-distance, 2: 2D tangent-plane
        residual_order: float = 0.5,
) -> tuple[np.array, np.array, np.array, np.array, np.array, int, float]:
    """Second-step optimization method for initial relative orbit determination"""
    DIM = 6
    start = time.time()
    """Generate polynomials"""
    if polynomials is None:
        polynomials = generate_polynomials(
            initial_state=xc0,
            t_series=t_series,
            ft=ft,
            order=order,
            if_push_order=False,
        )
    time_cost = time.time() - start
    reference_point = initial_guess.copy()
    """Begin recursive optimization"""
    tol = np.inf
    iteration = 0
    flag = 1
    solutions_history = np.zeros([max_iteration, DIM])
    Maps = polynomials.copy()
    while tol > eps:
        with DA.cache_manager():  # optional, for efficiency
            for k in range(len(polynomials)):
                Maps[k] = polynomials[k].eval(reference_point + array.identity(DIM))
        """Generate STMs and STTs"""
        start = time.time()
        STMs = np.zeros([len(t_series), DIM, DIM])
        for k in range(len(t_series)):
            STM, _ = get_stm_stt_coefficients(
                polynomial=Maps[k],
                DIM=DIM,
                order=1,
            )
            STMs[k] = STM
        """Implement recursive optimization algorithm"""
        if if_add_weights is False:
            dx0 = RIOD_convex_optimization(
                STMs=STMs,
                Maps=Maps,
                los_vectors=los_vectors,
                ifPrint=ifPrint,
                if_first_order_cost=if_first_order_cost,
                residual_order=residual_order,
            )
        else:
            dx0 = RIOD_convex_optimization_weighted(
                STMs=STMs,
                Maps=Maps,
                los_vectors=los_vectors,
                R=R,
                weight_strategy=weight_strategy,
                ifPrint=ifPrint,
                if_first_order_cost=if_first_order_cost,
            )
        time_cost += time.time() - start
        """Update solutions"""
        reference_point += dx0
        solutions_history[iteration] = reference_point
        iteration += 1
        tol = np.linalg.norm(dx0)
        if iteration >= max_iteration:
            flag = 0
            break
    """Generate solutions"""
    solution = xc0 + reference_point
    true_states = np.concatenate((xd0, (xd0 - xc0)))
    AE = abs(np.concatenate((solution, reference_point)) - true_states)
    RE = np.where(true_states == 0, np.nan, AE / np.abs(true_states) * 100)
    """Return results"""
    dx0 = reference_point
    return dx0, solution, AE, RE, solutions_history, flag, time_cost

def adaptive_recursive_optimization_method(
        xc0: np.array,
        xd0: np.array,
        R: np.array,
        t_series: np.array,
        los_vectors: np.array,
        ft: Callable[[array, float], array],
        order: int,
        polynomials: list = None,
        max_iteration: int = 100,
        eps: float = 1e-6,
        constraints_bound: np.array = np.array([1e-4, 3e-3, 1e-1]),
        ifPrint: bool = True,
        if_first_order_cost: bool = True,
        if_add_weights: bool = False,
        weight_strategy: int = 0,  # 0: original OLOD, 1: inverse-distance, 2: 2D tangent-plane
        residual_order: float = 0.5,
) -> tuple[np.array, np.array, np.array, np.array, np.array, int, np.array, float, np.array]:
    """Adaptive recursive optimization method for initial relative orbit determination"""

    def generate_guess_for_bad_case(
            residuals: np.array,
            solutions_history: np.array,
    ):
        if np.min(residuals) >= 1e6:
            dx02 = solutions_history[0, 0]
        else:
            iter_min = np.argmin(residuals)
            dx02 = solutions_history[iter_min, 0]
        return dx02

    """Adaptively adjust the minimal distance for avoiding the convergence in a undesired solution"""
    DIM = 6
    time_costs = np.zeros(2)
    start = time.time()
    """Generate polynomials"""
    if polynomials is None:
        polynomials = generate_polynomials(
            initial_state=xc0,
            t_series=t_series,
            ft=ft,
            order=order,
            if_push_order=False,
        )
    time_costs[0] = time.time() - start
    """Generate the initial guess"""
    initial_guess = np.zeros(DIM)
    initial_guess[:3] = los_vectors[0] * constraints_bound[0]
    """Begin adaptive and recursive optimization"""
    minimal_distance = constraints_bound[0]
    flag = 1
    iteration = 0
    solutions_history = np.zeros([max_iteration, 2, DIM])
    residuals = np.ones([max_iteration]) * 1e8
    while True:
        """First-step optimization"""
        dx01, _, _, _, _, _, time_cost = first_step_optimization(
            xc0=xc0,
            xd0=xd0,
            R=R,
            initial_guess=initial_guess,
            t_series=t_series,
            los_vectors=los_vectors,
            ft=ft,
            order=order,
            polynomials=polynomials,
            minimal_distance=minimal_distance,
            max_iteration=max_iteration,
            eps=eps,
            ifPrint=ifPrint,
            if_first_order_cost=if_first_order_cost,
            if_add_weights=if_add_weights,
            weight_strategy=weight_strategy,
            residual_order=residual_order,
        )
        time_costs[1] += time_cost
        solutions_history[iteration, 0] = dx01
        """Cross-product tests"""
        if np.linalg.norm(dx01) >= math.sqrt(3) * 1e-1:
            residual = 1e8
        else:
            residual = 0.0
            relative_pos = np.zeros((len(t_series), 3))
            for i in range(len(t_series)):
                relative_pos[i] = polynomials[i].eval(dx01)[:3]
                residual += np.linalg.norm(los_vectors[i] - relative_pos[i] / np.linalg.norm(relative_pos[i]))
        residuals[iteration] = residual
        """Second-step optimization"""
        dx02, _, _, _, _, _, time_cost = second_step_optimization(
            xc0=xc0,
            xd0=xd0,
            R=R,
            initial_guess=dx01,
            t_series=t_series,
            los_vectors=los_vectors,
            ft=ft,
            order=order,
            polynomials=polynomials,
            max_iteration=max_iteration,
            eps=eps,
            ifPrint=ifPrint,
            if_first_order_cost=if_first_order_cost,
            if_add_weights=if_add_weights,
            weight_strategy=weight_strategy,
            residual_order=residual_order,
        )
        time_costs[1] += time_cost
        solutions_history[iteration, 1] = dx02
        """Adaptively adjust the constraints"""
        iteration += 1
        if iteration >= max_iteration:
            flag = 0
            dx02 = generate_guess_for_bad_case(
                residuals=residuals,
                solutions_history=solutions_history,
            )
            break
        if np.linalg.norm(dx02[:3]) <= 1e-4 or np.linalg.norm(dx02[:3]) >= math.sqrt(3) * 1e-1:
            """Converge into the false solution"""
            if minimal_distance <= constraints_bound[-1]:
                minimal_distance *= 2.0
                """Update the initial guess"""
                # initial_guess = dx01
                initial_guess = np.zeros(DIM)
                initial_guess[:3] = los_vectors[0] * minimal_distance
            else:
                """Fail to converge"""
                flag = 0
                dx02 = generate_guess_for_bad_case(
                    residuals=residuals,
                    solutions_history=solutions_history, 
                )
                break
        else:
            """Converge into the desired solution"""
            break
    """Generate solutions"""
    dx0 = dx02
    solution = xc0 + dx0
    true_states = np.concatenate((xd0, (xd0 - xc0)))
    AE = abs(np.concatenate((solution, dx0)) - true_states)
    RE = np.where(true_states == 0, np.nan, AE / np.abs(true_states) * 100)
    """Return results"""
    return dx0, solution, AE, RE, solutions_history, flag, time_costs, minimal_distance, residuals

if __name__ == "__main__":
    """Main function"""
    example_run()

