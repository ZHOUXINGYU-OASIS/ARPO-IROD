import sys
import mosek
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy import sparse
from scipy.stats import chi2
from daceypy import DA, array, ADS
from scipy.optimize import minimize
from module_measurements import skew_symmetric_matrix, handle_singular_weight_matrix

default_std = 1e-4

def streamprinter(text):
    """
    Define a stream printer to grab output from MOSEK
    """
    sys.stdout.write(text)
    sys.stdout.flush()

def RIOD_convex_optimization(
        STMs: np.array,
        Maps: list,
        los_vectors: np.array,
        ifPrint: bool = False,
        if_first_order_cost: bool = True,
        residual_order: float = 0.5,
) -> np.array:
    """Solve for the closest point using the convex optimization"""
    num_observations = len(los_vectors)
    DIM = 6
    DIMz = 3
    """Implement convex optimization"""
    with mosek.Task() as task:
        inf = 0.0  # Since the value of infinity is ignored, we define it solely for symbolic purposes
        """Attach a log stream printer to the task"""
        if ifPrint == 1:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        else:
            task.set_Stream(mosek.streamtype.log, None)
        numvar, numcon = (DIM + num_observations), 0
        task.appendvars(numvar)
        task.appendcons(numcon)
        """Add objective function"""
        csub = range(DIM, DIM + num_observations)
        cval = np.ones(num_observations)
        task.putclist(csub, cval)
        task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [inf] * numvar, [inf] * numvar)  # Setting bounds on variables
        """Add conic constraints"""
        if if_first_order_cost is True:
            """Use first-order cost function"""
            prob_f = np.zeros([num_observations * (DIMz + 1), numvar])
            prob_g = np.zeros([num_observations * (DIMz + 1), 1])
            for k in range(num_observations):
                los_cross = skew_symmetric_matrix(vector=los_vectors[k])
                x = Maps[k].cons()  # mean state
                A = STMs[k][:3, :]
                f = los_cross @ A
                g = los_cross @ (x[:3].reshape(DIMz, 1))
                prob_f[k * (DIMz + 1), DIM + k] = 1
                prob_f[(k * (DIMz + 1) + 1): ((k + 1) * (DIMz + 1)), :DIM] = f
                prob_g[(k * (DIMz + 1) + 1): ((k + 1) * (DIMz + 1)), :] = g
            prob_f = sparse.coo_matrix(prob_f)
            prob_g = sparse.coo_matrix(prob_g)
            task.appendafes(num_observations * (DIMz + 1))
            task.putafefentrylist(prob_f.row,  # Rows
                                  prob_f.col,  # Columns
                                  prob_f.data)
            for i in range(len(prob_g.row)):
                task.putafeg(prob_g.row[i], prob_g.data[i])
            for k in range(num_observations):
                task.appendacc(task.appendquadraticconedomain(DIMz + 1),  # Domain
                               np.arange(k * (DIMz + 1), (k + 1) * (DIMz + 1)),  # Rows from F
                               None)  # Unused
        else:
            """Use second-order cost function"""
            prob_f = np.zeros([num_observations * (DIMz + 2), numvar])
            prob_g = np.zeros([num_observations * (DIMz + 2), 1])
            for k in range(num_observations):
                los_cross = skew_symmetric_matrix(vector=los_vectors[k])
                x = Maps[k].cons()  # mean state
                A = STMs[k][:3, :]
                f = los_cross @ A
                g = los_cross @ (x[:3].reshape(DIMz, 1))
                prob_f[k * (DIMz + 2), DIM + k] = 1
                prob_g[k * (DIMz + 2) + 1, 0] = 1
                prob_f[(k * (DIMz + 2) + 2): ((k + 1) * (DIMz + 2)), :DIM] = f
                prob_g[(k * (DIMz + 2) + 2): ((k + 1) * (DIMz + 2)), :] = g
            prob_f = sparse.coo_matrix(prob_f)
            prob_g = sparse.coo_matrix(prob_g)
            task.appendafes(num_observations * (DIMz + 2))
            task.putafefentrylist(prob_f.row,  # Rows
                                  prob_f.col,  # Columns
                                  prob_f.data)
            for i in range(len(prob_g.row)):
                task.putafeg(prob_g.row[i], prob_g.data[i])
            for k in range(num_observations):
                task.appendacc(task.appendprimalpowerconedomain(DIMz + 2, [residual_order, 1.0 - residual_order]),  # Domain
                               np.arange(k * (DIMz + 2), (k + 1) * (DIMz + 2)),  # Rows from F
                               None)  # Unused
        """Input the objective sense (minimize/maximize)"""
        task.putobjsense(mosek.objsense.minimize)
        """Optimize the task"""
        task.optimize()
        sol = task.getxx(mosek.soltype.itr)
        sol = np.array(sol[:DIM])
    """Return results"""
    return sol

def RIOD_convex_optimization_weighted(
        STMs: np.array,
        Maps: list,
        los_vectors: np.array,
        R: np.array,
        ifPrint: bool = False,
        if_first_order_cost: bool = True,
        residual_order: float = 0.5,
) -> np.array:
    """Solve for the closest point using the convex optimization (using weighting strategy)"""
    num_observations = len(los_vectors)
    DIM = 6
    DIMz = 3
    """Add weights"""
    W = np.zeros((num_observations, DIMz, DIMz))
    U = np.zeros((num_observations, DIMz, DIMz))
    los_cross = np.zeros((num_observations, DIMz, DIMz))
    x = np.zeros((num_observations, DIM))
    A = np.zeros((num_observations, DIMz, DIM))
    for k in range(num_observations):
        los_cross[k] = skew_symmetric_matrix(vector=los_vectors[k])
        x[k] = Maps[k].cons()  # mean state
        A[k] = STMs[k][:3, :]
        """Determine the coefficients"""
        wc = skew_symmetric_matrix(vector=x[k][:3])
        try:
            W[k] = handle_singular_weight_matrix(wc @ R[k] @ wc.T)
            U[k] = cholesky(np.linalg.inv(W[k]), lower=False)
        except:
            W[k] = handle_singular_weight_matrix(wc @ R[k] @ wc.T, 1e-10)
            W[k] += 1e-12 * np.eye(W[k].shape[0])
            U[k] = cholesky(np.linalg.inv(W[k]), lower=False)
    """Implement convex optimization"""
    with mosek.Task() as task:
        inf = 0.0  # Since the value of infinity is ignored, we define it solely for symbolic purposes
        """Attach a log stream printer to the task"""
        if ifPrint == 1:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        else:
            task.set_Stream(mosek.streamtype.log, None)
        numvar, numcon = (DIM + num_observations), 0
        task.appendvars(numvar)
        task.appendcons(numcon)
        """Add objective function"""
        csub = range(DIM, DIM + num_observations)
        cval = np.ones(num_observations)
        task.putclist(csub, cval)
        task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [inf] * numvar, [inf] * numvar)  # Setting bounds on variables
        """Add conic constraints"""
        if if_first_order_cost is True:
            """Use first-order cost function"""
            prob_f = np.zeros([num_observations * (DIMz + 1), numvar])
            prob_g = np.zeros([num_observations * (DIMz + 1), 1])
            for k in range(num_observations):
                f = U[k] @ (los_cross[k] @ A[k])
                g = U[k] @ (los_cross[k] @ (x[k][:3].reshape(DIMz, 1)))
                prob_f[k * (DIMz + 1), DIM + k] = 1
                prob_f[(k * (DIMz + 1) + 1): ((k + 1) * (DIMz + 1)), :DIM] = f
                prob_g[(k * (DIMz + 1) + 1): ((k + 1) * (DIMz + 1)), :] = g
            prob_f = sparse.coo_matrix(prob_f)
            prob_g = sparse.coo_matrix(prob_g)
            task.appendafes(num_observations * (DIMz + 1))
            task.putafefentrylist(prob_f.row,  # Rows
                                  prob_f.col,  # Columns
                                  prob_f.data)
            for i in range(len(prob_g.row)):
                task.putafeg(prob_g.row[i], prob_g.data[i])
            for k in range(num_observations):
                task.appendacc(task.appendquadraticconedomain(DIMz + 1),  # Domain
                               np.arange(k * (DIMz + 1), (k + 1) * (DIMz + 1)),  # Rows from F
                               None)  # Unused
        else:
            """Use second-order cost function"""
            prob_f = np.zeros([num_observations * (DIMz + 2), numvar])
            prob_g = np.zeros([num_observations * (DIMz + 2), 1])
            for k in range(num_observations):
                f = U[k] @ (los_cross[k] @ A[k])
                g = U[k] @ (los_cross[k] @ (x[k][:3].reshape(DIMz, 1)))
                prob_f[k * (DIMz + 2), DIM + k] = 1
                prob_g[k * (DIMz + 2) + 1, 0] = 1
                prob_f[(k * (DIMz + 2) + 2): ((k + 1) * (DIMz + 2)), :DIM] = f
                prob_g[(k * (DIMz + 2) + 2): ((k + 1) * (DIMz + 2)), :] = g
            prob_f = sparse.coo_matrix(prob_f)
            prob_g = sparse.coo_matrix(prob_g)
            task.appendafes(num_observations * (DIMz + 2))
            task.putafefentrylist(prob_f.row,  # Rows
                                  prob_f.col,  # Columns
                                  prob_f.data)
            for i in range(len(prob_g.row)):
                task.putafeg(prob_g.row[i], prob_g.data[i])
            for k in range(num_observations):
                task.appendacc(task.appendprimalpowerconedomain(DIMz + 2, [residual_order, 1.0 - residual_order]),  # Domain
                               np.arange(k * (DIMz + 2), (k + 1) * (DIMz + 2)),  # Rows from F
                               None)  # Unused
        """Input the objective sense (minimize/maximize)"""
        task.putobjsense(mosek.objsense.minimize)
        """Optimize the task"""
        task.optimize()
        sol = task.getxx(mosek.soltype.itr)
        sol = np.array(sol[:DIM])
    """Return results"""
    return sol

if __name__ == "__main__":
    """Main function"""
    # 构造对称正定矩阵 W
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
    # A = np.eye(3)
    # 计算 Cholesky 分解 (下三角)
    L = cholesky(A, lower=True)

    print("W 矩阵:\n", A)
    print("\nCholesky 分解的 L:\n", L)
    print("\n验证 W = L @ L.T:\n", L @ L.T)

