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
from module_odea import pseudo_inverse_RIOD, quadratic_eigenvalue_RIOD
from module_tbp import TBP, TBP_dynamics
from module_measurements import process_los_with_noise, generate_polynomials, get_stm_stt_coefficients, \
    skew_symmetric_matrix, mahalanobis_distance, spectral_norm_upper_left_3x3
from module_optimization import RIOD_convex_optimization, RIOD_convex_optimization_weighted
from module_constrained_optimization import RIOD_convex_constrained_optimization, \
    RIOD_convex_constrained_optimization_weighted

warnings.filterwarnings("ignore")
RelTol = 1e-12
AbsTol = 1e-12
default_std = 1e-4

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_flyby_scenario(
        num_observations: int = 10,
        std: float = default_std,
        if_add_noises: bool = True,
        factor: float = 10.0,
        revolution: float = 1.0,
        if_set_seed: bool = True,
        seed: int = 42,
) -> tuple[np.array, np.array, np.array, np.array, Callable[[float, np.array], np.array], Callable[[array, float], array], np.array, np.array, dict]:
    """Get the scenario for the TBP case"""
    DIMz = 3
    if if_set_seed is True:
        setup_seed(seed)
    """Define initial states"""
    xc0 = np.array([1, 0, 0, 0, 1, 0])
    xd0 = xc0 + np.array([0.001, 0.001, 0, 0.001, 0, 0]) * factor
    """Define dynamics models"""
    f = lambda t, x: TBP_dynamics(t, x, miu=1.0)
    ft = lambda x, tau: TBP(x, tau, miu=1.0)
    """Propagate the nominal orbits"""
    t0 = 0.0
    tf = 2.0 * math.pi * revolution
    t_eval = np.linspace(t0, tf, num_observations)
    chief = solve_ivp(f, [t0, tf], xc0, args=(), method='RK45',
                    t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    chief_states = chief.y.T
    deputy = solve_ivp(f, [t0, tf], xd0, args=(), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    deputy_states = deputy.y.T
    """Generate the LOS measurements"""
    t_series = t_eval
    los_vectors = (deputy_states - chief_states)[:, 0:3]
    if if_add_noises is True:
        std_dev = std
    else:
        std_dev = 0.0
    R = {
        "LOSr": np.zeros((num_observations, 3, 3)),  # virtual LOS measurement (without noises)
        "LOS": np.zeros((num_observations, 3, 3)),
    }
    for k in range(len(t_series)):
        los_vectors[k] = los_vectors[k] / np.linalg.norm(los_vectors[k])
        los = los_vectors[k].reshape(DIMz, 1)
        R["LOSr"][k] = 1 / 2 * (std_dev ** 2) * (np.eye(DIMz) - los @ los.T)
        los_vectors[k] += np.random.multivariate_normal(
            mean=np.zeros(DIMz),
            cov=R["LOSr"][k],
            size=1,
        )[0]
        los = los_vectors[k].reshape(DIMz, 1)
        R["LOS"][k] = 1 / 2 * (std_dev ** 2) * (np.eye(DIMz) - los @ los.T)
    """Return results"""
    return xc0, xd0, t_series, los_vectors, f, ft, chief_states, deputy_states, R

def get_formation_scenario(
        num_observations: int = 10,
        std: float = default_std,
        if_add_noises: bool = True,
        factor: float = 1.0,
        revolution: float = 1.0,
        if_set_seed: bool = True,
        seed: int = 42,
) -> tuple[np.array, np.array, np.array, np.array, Callable[[float, np.array], np.array], Callable[[array, float], array], np.array, np.array, np.array]:
    """Get the scenario for the CRTBP case"""
    DIMz = 3
    if if_set_seed is True:
        setup_seed(seed)
    """Define initial states"""
    xc0 = np.array([1, 0, 0, 0, 1, 0])
    xd0 = xc0 + np.array([0, 0.001, 0, 0, 0, 0]) * factor
    """Define dynamics models"""
    f = lambda t, x: TBP_dynamics(t, x, miu=1.0)
    ft = lambda x, tau: TBP(x, tau, miu=1.0)
    """Propagate the nominal orbits"""
    t0 = 0.0
    tf = 2.0 * math.pi * revolution
    t_eval = np.linspace(t0, tf, num_observations)
    chief = solve_ivp(f, [t0, tf], xc0, args=(), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    chief_states = chief.y.T
    deputy = solve_ivp(f, [t0, tf], xd0, args=(), method='RK45',
                       t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    deputy_states = deputy.y.T
    """Generate the LOS measurements"""
    t_series = t_eval
    los_vectors = (deputy_states - chief_states)[:, 0:3]
    if if_add_noises is True:
        std_dev = std
    else:
        std_dev = 0.0
    R = {
        "LOSr": np.zeros((num_observations, 3, 3)),  # virtual LOS measurement (without noises)
        "LOS": np.zeros((num_observations, 3, 3)),
    }
    for k in range(len(t_series)):
        los_vectors[k] = los_vectors[k] / np.linalg.norm(los_vectors[k])
        los = los_vectors[k].reshape(DIMz, 1)
        R["LOSr"][k] = 1 / 2 * (std_dev ** 2) * (np.eye(DIMz) - los @ los.T)
        los_vectors[k] += np.random.multivariate_normal(
            mean=np.zeros(DIMz),
            cov=R["LOSr"][k],
            size=1,
        )[0]
        los = los_vectors[k].reshape(DIMz, 1)
        R["LOS"][k] = 1 / 2 * (std_dev ** 2) * (np.eye(DIMz) - los @ los.T)
    """Return results"""
    return xc0, xd0, t_series, los_vectors, f, ft, chief_states, deputy_states, R

def get_general_scenario(
        xc0: np.array,
        xd0: np.array,
        num_observations: int = 10,
        std: float = default_std,
        if_add_noises: bool = True,
        revolution: float = 1.0,
        if_set_seed: bool = True,
        seed: int = 42,
        los_noiese: np.array = None,
) -> tuple[np.array, np.array, np.array, np.array, Callable[[float, np.array], np.array], Callable[[array, float], array], np.array, np.array, np.array]:
    """Get the scenario for the CRTBP case"""
    DIMz = 3
    if if_set_seed is True:
        setup_seed(seed)
    """Define dynamics models"""
    f = lambda t, x: TBP_dynamics(t, x, miu=1.0)
    ft = lambda x, tau: TBP(x, tau, miu=1.0)
    """Propagate the nominal orbits"""
    t0 = 0.0
    tf = 2.0 * math.pi * revolution
    t_eval = np.linspace(t0, tf, num_observations)
    chief = solve_ivp(f, [t0, tf], xc0, args=(), method='RK45',
                      t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    chief_states = chief.y.T
    deputy = solve_ivp(f, [t0, tf], xd0, args=(), method='RK45',
                       t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    deputy_states = deputy.y.T
    """Generate the LOS measurements"""
    t_series = t_eval
    los_vectors = (deputy_states - chief_states)[:, 0:3]
    if if_add_noises is True:
        std_dev = std
    else:
        std_dev = 0.0
    R = {
        "LOSr": np.zeros((num_observations, 3, 3)),  # virtual LOS measurement (without noises)
        "LOS": np.zeros((num_observations, 3, 3)),
    }
    for k in range(len(t_series)):
        los_vectors[k] = los_vectors[k] / np.linalg.norm(los_vectors[k])
        los = los_vectors[k].reshape(DIMz, 1)
        R["LOSr"][k] = 1 / 2 * (std_dev ** 2) * (np.eye(DIMz) - los @ los.T)
        if los_noiese is None:
            los_vectors[k] += np.random.multivariate_normal(
                mean=np.zeros(DIMz),
                cov=R["LOSr"][k],
                size=1,
            )[0]
        else:
            los_vectors[k] += los_noiese[k]  # use this part of code if parallel computation is employed
        los = los_vectors[k].reshape(DIMz, 1)
        R["LOS"][k] = 1 / 2 * (std_dev ** 2) * (np.eye(DIMz) - los @ los.T)
    """Return results"""
    return xc0, xd0, t_series, los_vectors, f, ft, chief_states, deputy_states, R

def pseudo_inverse_method(
        xc0: np.array,
        xd0: np.array,
        t_series: np.array,
        los_vectors: np.array,
        ft: Callable[[array, float], array],
        order: int,
        if_push_order: bool = True,
) -> tuple[np.array, np.array, np.array, np.array, float, list]:
    """Pseudo inverse initial relative orbit determination"""
    DIM = 6
    start = time.time()
    """Generate polynomials"""
    polynomials = generate_polynomials(
        initial_state=xc0,
        t_series=t_series,
        ft=ft,
        order=order,
        if_push_order=if_push_order,
    )
    """Generate STMs and STTs"""
    STMs = np.zeros([len(t_series), DIM, DIM])
    STTs = np.zeros([len(t_series), DIM, DIM, DIM])
    for k in range(len(t_series)):
        STM, STT = get_stm_stt_coefficients(
            polynomial=polynomials[k],
            DIM=DIM,
        )
        STMs[k] = STM
        STTs[k] = STT
    STMs = STMs[:, 0:3, :]  # remove derivatives in STMs and STTs
    STTs = STTs[:, 0:3, :, :]
    """Implement pseudo inverse algorithm"""
    dx0 = pseudo_inverse_RIOD(los_vectors, STMs, STTs)
    solution = xc0 + dx0
    AE = abs(np.concatenate((solution, dx0)) - np.concatenate((xd0, (xd0 - xc0))))
    RE = AE / abs(np.concatenate((
        xd0 + np.ones(DIM) * 1e-10,
        (xd0 - xc0) + np.ones(DIM) * 1e-10)
    ))
    """Return results"""
    time_cost = time.time() - start
    return dx0, solution, AE, RE, time_cost, polynomials

def quadratic_eigenvalue_method(
        xc0: np.array,
        xd0: np.array,
        t_series: np.array,
        los_vectors: np.array,
        ft: Callable[[array, float], array],
        order: int,
        if_push_order: bool = True,
) -> tuple[np.array, np.array, np.array, np.array, float, list]:
    """Pseudo inverse initial relative orbit determination"""
    DIM = 6
    start = time.time()
    """Generate polynomials"""
    polynomials = generate_polynomials(
        initial_state=xc0,
        t_series=t_series,
        ft=ft,
        order=order,
        if_push_order=if_push_order,
    )
    """Generate STMs and STTs"""
    STMs = np.zeros([len(t_series), DIM, DIM])
    STTs = np.zeros([len(t_series), DIM, DIM, DIM])
    for k in range(len(t_series)):
        STM, STT = get_stm_stt_coefficients(
            polynomial=polynomials[k],
            DIM=DIM,
        )
        STMs[k] = STM
        STTs[k] = STT
    STMs = STMs[:, 0:3, :]  # remove derivatives in STMs and STTs
    STTs = STTs[:, 0:3, :, :]
    """Implement quadratic eigenvalue algorithm"""
    dx0 = quadratic_eigenvalue_RIOD(los_vectors, STMs, STTs)
    solution = xc0 + dx0
    AE = abs(np.concatenate((solution, dx0)) - np.concatenate((xd0, (xd0 - xc0))))
    RE = AE / abs(np.concatenate((
        xd0 + np.ones(DIM) * 1e-10,
        (xd0 - xc0) + np.ones(DIM) * 1e-10)
    ))
    """Return results"""
    time_cost = time.time() - start
    return dx0, solution, AE, RE, time_cost, polynomials

def recursive_optimization_method(
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
        if_add_constraints: bool = False,
        minimal_distance: float = 1e-3,
        residual_order: float = 0.5,
) -> tuple[np.array, np.array, np.array, np.array, np.array, int, float]:
    """Pseudo inverse initial relative orbit determination"""
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
        if if_add_constraints is False:
            """Unconstrained optimization"""
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
                    ifPrint=ifPrint,
                    if_first_order_cost=if_first_order_cost,
                )
        else:
            """Constrained optimization"""
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

if __name__ == "__main__":
    """Main function"""
    order = 5
    default_std = 1e-4
    xc0, xd0, t_series, los_vectors, f, ft, chief_states, deputy_states, R = get_flyby_scenario(
        num_observations=10,
        factor=10.0,
        revolution=1.0,
        std=default_std,
        if_add_noises=True,
        if_set_seed=True,
    )
    # xc0, xd0, t_series, los_vectors, f, ft, chief_states, deputy_states, R = get_formation_scenario(
    #     num_observations=10,
    #     factor=1.0,
    #     revolution=1.0,
    #     std=default_std,
    #     if_add_noises=True,
    #     if_set_seed=True,
    # )
    """Pseudo inverse algorithm"""
    dx0, solution1, AE1, RE1, time_cost1, polynomials = pseudo_inverse_method(
        xc0=xc0,
        xd0=xd0,
        t_series=t_series,
        los_vectors=los_vectors,
        ft=ft,
        order=order,
    )
    """Quadratic eigenvalue algorithm"""
    # dx0, solution1, AE1, RE1, time_cost1, _ = quadratic_eigenvalue_method(
    #     xc0=xc0,
    #     xd0=xd0,
    #     t_series=t_series,
    #     los_vectors=los_vectors,
    #     ft=ft,
    #     order=order,
    # )
    """Convex optimization method"""
    initial_guess = dx0
    """Without the constraint"""
    # dx02, solution2, AE2, RE2, _, flag2, time_cost2 = recursive_optimization_method(
    #     xc0=xc0,
    #     xd0=xd0,
    #     R=R["LOS"],
    #     initial_guess=initial_guess,
    #     t_series=t_series,
    #     los_vectors=los_vectors,
    #     ft=ft,
    #     order=order,
    #     polynomials=polynomials,
    #     max_iteration=100,
    #     eps=1e-8,
    #     ifPrint=True,
    #     if_first_order_cost=True,
    #     if_add_weights=False,
    #     if_add_constraints=False,
    # )
    """With the constraint"""
    # dx03, solution3, AE3, RE3, _, flag3, time_cost3 = recursive_optimization_method(
    #     xc0=xc0,
    #     xd0=xd0,
    #     R=R["LOS"],
    #     initial_guess=dx02,
    #     t_series=t_series,
    #     los_vectors=los_vectors,
    #     ft=ft,
    #     order=order,
    #     polynomials=polynomials,
    #     max_iteration=100,
    #     eps=1e-8,
    #     ifPrint=True,
    #     if_first_order_cost=True,
    #     if_add_weights=False,
    #     if_add_constraints=True,
    # )
    """Without the weights"""
    dx02, solution2, AE2, RE2, _, flag2, time_cost2 = recursive_optimization_method(
        xc0=xc0,
        xd0=xd0,
        R=R["LOS"],
        initial_guess=dx0,
        t_series=t_series,
        los_vectors=los_vectors,
        ft=ft,
        order=order,
        polynomials=polynomials,
        max_iteration=100,
        eps=1e-8,
        ifPrint=False,
        if_first_order_cost=True,
        if_add_weights=False,
        if_add_constraints=False,
    )
    """With the weights"""
    dx03, solution3, AE3, RE3, _, flag3, time_cost3 = recursive_optimization_method(
        xc0=xc0,
        xd0=xd0,
        R=R["LOS"],
        initial_guess=dx0,
        t_series=t_series,
        los_vectors=los_vectors,
        ft=ft,
        order=order,
        polynomials=polynomials,
        max_iteration=100,
        eps=1e-8,
        ifPrint=False,
        if_first_order_cost=True,
        if_add_weights=True,
        if_add_constraints=False,
    )
