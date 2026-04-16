import numpy as np
import mosek
from scipy import sparse
from scipy.linalg import cholesky
from module_measurements import skew_symmetric_matrix, handle_singular_weight_matrix
from module_optimization import streamprinter

def normalize_vector(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Normalize a vector."""
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-norm vector encountered.")
    return v / n


def build_los_tangent_basis(los_vector: np.ndarray) -> np.ndarray:
    """
    Build a 3x2 orthonormal basis E = [e1, e2] for the LOS tangent plane.
    It satisfies E.T @ los = 0 and E.T @ E = I.
    """
    l = normalize_vector(los_vector)

    if abs(l[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = ref - np.dot(ref, l) * l
    e1 = normalize_vector(e1)

    e2 = np.cross(l, e1)
    e2 = normalize_vector(e2)

    return np.column_stack((e1, e2))  # shape: (3, 2)

def make_spd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetrize and regularize a matrix to be SPD."""
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals[eigvals < eps] = eps
    M_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (M_spd + M_spd.T)

def RIOD_convex_constrained_optimization_weighted(
        guess: np.ndarray,
        STMs: np.ndarray,
        Maps: list,
        los_vectors: np.ndarray,
        R: np.ndarray,
        weight_strategy: int = 0,   # 0: original OLOD, 1: inverse-distance, 2: 2D tangent-plane
        ifPrint: bool = False,
        if_first_order_cost: bool = True,
        minimal_distance: float = 1e-3,
        residual_order: float = 0.5,
) -> np.ndarray:
    """
    Solve for the closest point using convex optimization with a zero-avoidance constraint.

    Parameters
    ----------
    guess : np.ndarray
        Initial guess for the solution, shape (6,).
    STMs : np.ndarray
        State transition matrices, shape (N, 6, 6).
    Maps : list
        Polynomial maps. Maps[k].cons() should return the mean state at epoch k, shape (6,).
    los_vectors : np.ndarray
        LOS unit vectors, shape (N, 3).
    R : np.ndarray
        LOS covariance matrices, shape (N, 3, 3).
    weight_strategy : int
        0 -> original OLOD weighting
        1 -> inverse-distance scalar weighting
        2 -> 2D tangent-plane weighting
    ifPrint : bool
        Whether to print Mosek logs.
    if_first_order_cost : bool
        If True, use standard SOC form.
        If False, use power-cone form.
    minimal_distance : float
        Minimum distance threshold in the linearized zero-avoidance constraint.
    residual_order : float
        Power-cone order if if_first_order_cost is False.

    Returns
    -------
    sol : np.ndarray
        Optimized 6D solution.
    """
    num_observations = len(los_vectors)
    DIM = 6

    # NOTE:
    # I keep your original setting DIMx = 6.
    # If your intention is to constrain only the position norm,
    # then DIMx should be changed to 3.
    DIMx = 6

    if weight_strategy not in [0, 1, 2]:
        raise ValueError("weight_strategy must be 0, 1, or 2.")

    # Residual dimension
    DIMz = 2 if weight_strategy == 2 else 3

    # Storage
    Q = np.zeros((num_observations, DIMz, DIMz))
    U = np.zeros((num_observations, DIMz, DIMz))
    los_op = np.zeros((num_observations, DIMz, 3))
    x = np.zeros((num_observations, DIM))
    Apos = np.zeros((num_observations, 3, DIM))
    E_basis = None
    if weight_strategy == 2:
        E_basis = np.zeros((num_observations, 3, 2))

    # Build weighting-related quantities
    for k in range(num_observations):
        l_k = normalize_vector(los_vectors[k])
        x[k] = Maps[k].cons()
        Apos[k] = STMs[k][:3, :]
        r_mean = x[k][:3]
        r_norm = np.linalg.norm(r_mean)

        if weight_strategy == 0:
            # Original OLOD weighting
            # Q_k = [r_mean]_x R_k [r_mean]_x^T
            los_op[k] = skew_symmetric_matrix(vector=l_k)  # 3x3
            wc = skew_symmetric_matrix(vector=r_mean)
            Qk = wc @ R[k] @ wc.T
            Qk = make_spd(Qk, eps=1e-14)

            try:
                # Try the original covariance first
                Qinv = np.linalg.inv(Qk)
                U[k] = cholesky(Qinv, lower=False)
                Q[k] = Qk
            except Exception:
                # Only regularize if Cholesky / inverse fails
                Qk = make_spd(Qk, eps=1e-12)
                Q[k] = Qk
                U[k] = cholesky(np.linalg.inv(Qk), lower=False)

        elif weight_strategy == 1:
            # Inverse-distance scalar weighting
            los_op[k] = skew_symmetric_matrix(vector=l_k)  # 3x3

            weight_scalar = 1.0 / max(r_norm, 1e-12)

            # Equivalent covariance form: Q = (1 / weight_scalar) * I
            Qk = (1.0 / weight_scalar) * np.eye(DIMz)
            Q[k] = Qk
            U[k] = np.sqrt(weight_scalar) * np.eye(DIMz)

        elif weight_strategy == 2:
            # 2D tangent-plane weighting
            E = build_los_tangent_basis(l_k)  # 3x2
            E_basis[k] = E

            # Project 3D cross-product residual onto tangent plane:
            # r_2d = E^T [l]_x delta_r
            L3 = skew_symmetric_matrix(vector=l_k)  # 3x3
            los_op[k] = E.T @ L3  # 2x3

            # Original OLOD covariance first in 3D, then projected to 2D
            wc = skew_symmetric_matrix(vector=r_mean)
            Q3 = wc @ R[k] @ wc.T
            Q2 = E.T @ Q3 @ E

            try:
                Qinv = np.linalg.inv(Q2)
                U[k] = cholesky(Qinv, lower=False)
                Q[k] = Q2
            except Exception:
                Q2 = make_spd(Q2, eps=1e-12)
                Qinv = np.linalg.inv(Q2)
                U[k] = cholesky(Qinv, lower=False)
                Q[k] = Q2

    # Implement convex optimization
    with mosek.Task() as task:
        inf = 0.0  # symbolic only

        # Log stream
        if ifPrint:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        else:
            task.set_Stream(mosek.streamtype.log, None)

        numvar, numcon = (DIM + num_observations), 1
        task.appendvars(numvar)
        task.appendcons(numcon)

        # Objective: minimize sum of auxiliary variables
        csub = range(DIM, DIM + num_observations)
        cval = np.ones(num_observations)
        task.putclist(csub, cval)

        # Free variables
        for k in range(numvar):
            task.putvarbound(k, mosek.boundkey.fr, -inf, +inf)

        # Linear zero-avoidance constraint
        d = np.linalg.norm(guess[:DIMx])
        if d < 1e-15:
            raise ValueError("The norm of guess[:DIMx] is too small for the linearized distance constraint.")

        prob_a = np.zeros((numcon, numvar))
        prob_a[0, :DIMx] = guess[:DIMx] / d
        prob_a = sparse.coo_matrix(prob_a)

        task.putaijlist(prob_a.row, prob_a.col, prob_a.data)
        task.putconbound(0, mosek.boundkey.lo, minimal_distance - d, +inf)

        # Conic constraints
        if if_first_order_cost:
            # Standard SOC: [t_k; U_k * residual_k] in Q^{DIMz+1}
            prob_f = np.zeros((num_observations * (DIMz + 1), numvar))
            prob_g = np.zeros((num_observations * (DIMz + 1), 1))

            for k in range(num_observations):
                f = U[k] @ (los_op[k] @ Apos[k])                      # (DIMz, 6)
                g = U[k] @ (los_op[k] @ x[k][:3].reshape(3, 1))      # (DIMz, 1)

                row0 = k * (DIMz + 1)
                prob_f[row0, DIM + k] = 1.0
                prob_f[row0 + 1: row0 + 1 + DIMz, :DIM] = f
                prob_g[row0 + 1: row0 + 1 + DIMz, :] = g

            prob_f = sparse.coo_matrix(prob_f)
            prob_g = sparse.coo_matrix(prob_g)

            task.appendafes(num_observations * (DIMz + 1))
            task.putafefentrylist(prob_f.row, prob_f.col, prob_f.data)

            for i in range(len(prob_g.row)):
                task.putafeg(prob_g.row[i], prob_g.data[i])

            for k in range(num_observations):
                row0 = k * (DIMz + 1)
                task.appendacc(
                    task.appendquadraticconedomain(DIMz + 1),
                    np.arange(row0, row0 + DIMz + 1),
                    None
                )

        else:
            # Power-cone: [t_k; 1; U_k * residual_k] in power cone
            prob_f = np.zeros((num_observations * (DIMz + 2), numvar))
            prob_g = np.zeros((num_observations * (DIMz + 2), 1))

            for k in range(num_observations):
                f = U[k] @ (los_op[k] @ Apos[k])                      # (DIMz, 6)
                g = U[k] @ (los_op[k] @ x[k][:3].reshape(3, 1))      # (DIMz, 1)

                row0 = k * (DIMz + 2)
                prob_f[row0, DIM + k] = 1.0
                prob_g[row0 + 1, 0] = 1.0
                prob_f[row0 + 2: row0 + 2 + DIMz, :DIM] = f
                prob_g[row0 + 2: row0 + 2 + DIMz, :] = g

            prob_f = sparse.coo_matrix(prob_f)
            prob_g = sparse.coo_matrix(prob_g)

            task.appendafes(num_observations * (DIMz + 2))
            task.putafefentrylist(prob_f.row, prob_f.col, prob_f.data)

            for i in range(len(prob_g.row)):
                task.putafeg(prob_g.row[i], prob_g.data[i])

            for k in range(num_observations):
                row0 = k * (DIMz + 2)
                task.appendacc(
                    task.appendprimalpowerconedomain(DIMz + 2, [residual_order, 1.0 - residual_order]),
                    np.arange(row0, row0 + DIMz + 2),
                    None
                )

        # Objective sense
        task.putobjsense(mosek.objsense.minimize)

        # Optimize
        task.optimize()

        sol = np.array(task.getxx(mosek.soltype.itr)[:DIM])

    return sol