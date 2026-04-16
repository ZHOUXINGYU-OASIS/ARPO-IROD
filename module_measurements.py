import numpy as np
import daceypy_import_helper  # noqa: F401
from typing import Callable
from daceypy import DA, array, ADS
from module_integrator import RK78

default_std = 1e-4

def spectral_norm_upper_left_3x3(covariance: np.ndarray) -> float:
    """
    Compute the 2-norm (spectral norm) of the upper-left 3x3 block of a covariance matrix.

    Parameters:
    covariance (np.ndarray): A 2D NumPy array representing the covariance matrix.

    Returns:
    float: The spectral norm (largest singular value) of the 3x3 block.
    """
    # Extract the upper-left 3x3 block
    upper_left_block = covariance[:3, :3]

    # Compute the spectral norm (largest singular value)
    norm_2 = np.linalg.norm(upper_left_block, ord=2)

    return norm_2

def mahalanobis_distance(
        error: np.array,
        covariance_matrix: np.array
) -> float:
    """
    Compute the Mahalanobis distance.

    Parameters:
    error (np.array): A one-dimensional numpy array representing the error vector.
    covariance_matrix (np.array): A two-dimensional numpy array representing the covariance matrix.

    Returns:
    float: The computed Mahalanobis distance.
    """
    # Compute the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(covariance_matrix)

    # Compute the Mahalanobis distance using the formula: sqrt(error.T * inv_cov_matrix * error)
    distance = np.sqrt(np.dot(np.dot(error.T, inv_cov_matrix), error))

    return float(distance)

def generate_polynomials(
        initial_state: np.array,
        t_series: np.array,
        ft: Callable[[array, float], array],
        order: int,
        if_push_order: bool = False,
        reduced_order: int = 2,
        if_preserve_constant: bool = False,
) -> list:
    """Generate polynomials"""
    """Initialize DA"""
    DIM = 6
    DA.init(order, DIM)
    DA.setEps(1e-32)
    if if_push_order is True:
        DA.pushTO(reduced_order)
    """Define DA variables"""
    x0 = initial_state + array.identity(DIM)
    """Propagate the orbits"""
    polynomials = list()
    if if_preserve_constant is True:
        polynomials.append(x0)
    else:
        polynomials.append(x0 - x0.cons())
    with DA.cache_manager():  # optional, for efficiency
        for k in range(len(t_series) - 1):
            t0 = t_series[k]
            tf = t_series[k + 1]
            xf = RK78(x0, t0, tf, ft)  # RK78 propagator
            if if_preserve_constant is True:
                polynomials.append(xf)
            else:
                polynomials.append(xf - xf.cons())
            x0 = xf.copy()
    """Return results"""
    return polynomials

def generate_los_polynomials(
        chief_states: np.array,
        t_series: np.array,
        polynomials: list,
) -> list:
    """Process the polynomials to generate the STM for the LOS measurements"""
    los_polynomials = list()
    """Generate the Taylor polynomials"""
    for k in range(len(t_series)):
        xd = polynomials[k]
        xc = chief_states[k]
        los_vector: array = (xd - xc)[0:3]
        los_vector: array = los_vector / los_vector.vnorm()
        los_polynomials.append(los_vector)
    """Return results"""
    return los_polynomials

def get_stm_stt_coefficients(
        polynomial: array,
        DIM: int = 6,
        order: int = 2,
) -> tuple[np.array, np.array]:
    """Get the STM and STT from the polynomials"""
    STM, STT = np.zeros([DIM, DIM]), np.zeros([DIM, DIM, DIM])
    """STM and STT"""
    with DA.cache_manager():  # optional, for efficiency
        for i in range(DIM):
            for j in range(DIM):
                STM[i, j] = polynomial[i].deriv(j + 1).cons()
                if order >= 2:
                    for k in range(DIM):
                        STT[i, j, k] = polynomial[i].deriv(j + 1).deriv(k + 1).cons()
    """Return data"""
    return STM, STT

def los_to_az_el(los):
    """
    Convert Line-of-Sight (LOS) vector to azimuth and elevation angles.
    :param los: A 3D numpy array representing the LOS vector.
    :return: Azimuth (radians), Elevation (radians)
    """
    x, y, z = los
    azimuth = np.arctan2(y, x)  # Calculate azimuth angle
    elevation = np.arcsin(z / np.linalg.norm(los))  # Calculate elevation angle
    return azimuth, elevation

def az_el_to_los(azimuth, elevation):
    """
    Convert azimuth and elevation angles back to LOS vector.
    :param azimuth: Azimuth angle in radians.
    :param elevation: Elevation angle in radians.
    :return: A 3D numpy array representing the LOS vector.
    """
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.array([x, y, z])

def add_gaussian_noise(value, std_dev):
    """
    Add Gaussian noise to a given value.
    :param value: The original value.
    :param std_dev: The standard deviation of the Gaussian noise.
    :return: The noisy value.
    """
    return value + np.random.normal(0.0, std_dev)

def process_los_with_noise(
        los: np.array,
        std_dev: float,
        R: np.array,
) -> tuple[np.array, np.array]:
    """
    Process a LOS vector by converting it to azimuth and elevation,
    applying Gaussian noise, and converting it back to a LOS vector.
    :param los: A 3D numpy array representing the LOS vector.
    :param std_dev: The standard deviation of the Gaussian noise in radians.
    :return: A new LOS vector with noise applied.
    """
    """Add noises"""
    azimuth, elevation = los_to_az_el(los)
    noisy_azimuth = add_gaussian_noise(azimuth, std_dev)
    noisy_elevation = add_gaussian_noise(elevation, std_dev)
    """Generate covariance"""
    Rlos = az_el_to_los_cov(
        azimuth=noisy_azimuth,
        elevation=noisy_elevation,
        cov_az_el=R,
    )
    return az_el_to_los(noisy_azimuth, noisy_elevation), Rlos


def skew_symmetric_matrix(vector):
    """
    Generate the skew-symmetric matrix of a 3D vector.

    The skew-symmetric matrix [v]x satisfies the cross-product operation:
        v × w = [v]x * w

    Parameters:
        vector (numpy array): A 3D vector [vx, vy, vz].

    Returns:
        numpy array: The 3×3 skew-symmetric matrix.
    """
    vx, vy, vz = vector
    return np.array([
        [0, -vz, vy],
        [vz, 0, -vx],
        [-vy, vx, 0]
    ])

def az_el_to_los_cov(
        azimuth: float,
        elevation: float,
        cov_az_el: np.array,
) -> np.array:
    """
    Converts the covariance matrix from azimuth-elevation representation to line-of-sight (LOS) vector representation.

    Parameters:
    azimuth  (float): Azimuth angle θ (radians)
    elevation (float): Elevation angle φ (radians)
    cov_az_el (2x2 ndarray): Covariance matrix of azimuth and elevation

    Returns:
    cov_r (3x3 ndarray): Covariance matrix of the LOS vector
    """

    theta = azimuth
    phi = elevation

    # Compute the Jacobian matrix J
    J = np.array([
        [-np.cos(phi) * np.sin(theta), -np.sin(phi) * np.cos(theta)],
        [ np.cos(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta)],
        [ 0, np.cos(phi)]
    ])

    # Covariance propagation: C_r = J * C_theta_phi * J^T
    cov_r = J @ cov_az_el @ J.T

    return cov_r

def handle_singular_weight_matrix(
        W: np.array,
        threshold: float = 1e-14,
) -> np.array:
    """The weight matrix is singular"""
    eigvals, eigvecs = np.linalg.eigh(W)
    eigvals[eigvals <= threshold] = threshold
    W_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return W_new