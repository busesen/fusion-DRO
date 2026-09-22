"""Coffee RKHS benchmark consolidated from the authors' code, with permission.

Xianyu Li, Huan Xu, Xiaolin Huang and Chao Shang (2026),
Robust Out-of-Distribution Stochastic Optimization, arXiv:2604.20147.
https://arxiv.org/abs/2604.20147

The coffee routines and adapter retain their existing mathematical formulation,
randomization, solver settings, stopping rules, and returned-status convention.
Only file organization and imports are changed; no external helper files are needed.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

DEFAULT_GUROBI_LICENSE = Path.home() / "gurobi.lic"
ALT_GUROBI_LICENSE_ENV = "IWB_GUROBI_LICENSE_FILE"


def check_rkhs_dependencies() -> None:
    """Import optional packages without opening a licensed solver session."""
    global gp, GRB, rbf_kernel, euclidean_distances, train_test_split, qmc
    try:
        import gurobipy as gp
        from gurobipy import GRB
        from sklearn.metrics.pairwise import rbf_kernel
        from sklearn.metrics import euclidean_distances
        from sklearn.model_selection import train_test_split
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError(
            "RKHS requires gurobipy and scikit-learn from requirements.txt. "
            "Install them or disable this benchmark with --run_rkhs False."
        ) from exc


def _ensure_gurobi_license(gurobi_license_file: str='') -> None:
    if str(gurobi_license_file).strip():
        os.environ['GRB_LICENSE_FILE'] = str(Path(gurobi_license_file).expanduser())
    elif not os.environ.get('GRB_LICENSE_FILE'):
        alternate = os.environ.get(ALT_GUROBI_LICENSE_ENV)
        if alternate:
            os.environ['GRB_LICENSE_FILE'] = str(Path(alternate).expanduser())
        elif DEFAULT_GUROBI_LICENSE.exists():
            os.environ['GRB_LICENSE_FILE'] = str(DEFAULT_GUROBI_LICENSE)

class rkhsFun:

    def __init__(self, kernelFun, gamma=None, casadi=False):
        """
        kernel gamma param
        """
        self.kernelFun = kernelFun
        self.kernel_gamma = gamma
        self.casadi = casadi

    def eval(self, x, a, data):
        """
        compute the rkhs fucntion
            f(x) = sum a_i * k(data_i, x)
        x: the location to evaluate, can be a vector
        data: the expansion points, empirical data
        """
        if len(x.shape) == 1:
            xloc = x.reshape(-1, 1)
        elif len(x.shape) == 2:
            xloc = x
        else:
            raise NotImplementedError
        k = self.kernelFun(data, xloc, gamma=self.kernel_gamma)
        fval = a @ k
        return fval

    def __call__(self, x, a, data):
        return self.eval(x, a, data)

def median_heuristic(X, Y):
    """
    the famous kernel median heuristic
    """
    dists = euclidean_distances(X, Y, squared=False)
    kernel_width = np.median(dists)
    'in sklearn, kernel is done by K(x, y) = exp(-gamma ||x-y||^2)'
    kernel_gamma = 1.0 / (2 * kernel_width ** 2)
    return (kernel_width, kernel_gamma)

def getKernel(X, kernel_gamma):
    """
    Compute the kernel matrix.
    """
    M = len(X)
    start_time = time.time()
    kernelMat = np.zeros((M, M))
    for m1 in range(M):
        for m2 in range(m1, M):
            X1 = X[m1]
            X2 = X[m2]
            K_block = rbf_kernel(X1, X2, gamma=kernel_gamma)
            kernelMat[m1, m2] = np.mean(K_block)
            kernelMat[m2, m1] = kernelMat[m1, m2]
    return kernelMat

def solvedual(kerMatrix, C, upper_bounds=None):
    (M, _) = np.shape(kerMatrix)
    if upper_bounds is None:
        caps = np.full(M, float(C))
    else:
        caps = np.asarray(upper_bounds, dtype=float).reshape(-1)
        if caps.shape[0] != M:
            raise ValueError(f'upper_bounds has length {caps.shape[0]} but kerMatrix is {M}x{M}')
        if np.any(caps < 0):
            raise ValueError('upper_bounds must be nonnegative')
    e = max(float(np.max(caps)), 1.0) * 1e-05
    m = gp.Model('dual')
    m.setParam('OutputFlag', 0)
    alpha = m.addMVar(shape=M, vtype=GRB.CONTINUOUS, lb=np.zeros(M), name='alpha')
    for i in range(M):
        m.addConstr(alpha[i] <= caps[i])
    m.addConstr(sum((alpha[i] for i in range(M))) == 1)
    m.setObjective(alpha.T @ kerMatrix @ alpha - alpha.T @ np.diag(kerMatrix), GRB.MINIMIZE)
    m.optimize()
    alpha_value = alpha.X
    SVindex = np.where(alpha_value >= e)[0]
    BSVindex = np.where((alpha_value >= e) & (alpha_value <= caps - e))[0]
    outlierindex = np.where(alpha_value > caps - e)[0]
    return (alpha.X, SVindex, BSVindex, outlierindex)

def kernel_herding(kernel, kernel_params, z_samples, T=50):
    """
    Kernel herding: construct T points from given samples using greedy selection.

    Parameters
    ----------
    z_samples : ndarray, shape (m, d)
        Initial empirical samples drawn from distribution P
    T : int
        Number of herding points to select
    kernel : str
        Kernel function to use ('rbf' currently supported)
    gamma : float
        Kernel parameter for RBF

    Returns
    -------
    x_list : list of ndarray
        List of selected herding points (shape T x d)
    """
    (m, d) = z_samples.shape
    x_list = []
    if kernel == 'Gaussian':
        (kernel_width, kernel_gamma) = kernel_params
        K_emp = rbf_kernel(z_samples, z_samples, gamma=kernel_gamma)
        mean_embedding = K_emp.mean(axis=1)
        scores = 2 * mean_embedding
        idx = np.argmax(scores)
        x_list.append(z_samples[idx])
        for t in range(1, T):
            herd_K = rbf_kernel(z_samples, np.array(x_list), gamma=kernel_gamma)
            avg_kernel = herd_K.mean(axis=1)
            scores = 2 * mean_embedding - t / (t + 1) * avg_kernel
            idx = np.argmax(scores)
            x_list.append(z_samples[idx])
    elif kernel == 'Laplace':
        pass
    return np.array(x_list)

def _safe_mvar_value(var, dim):
    try:
        return np.asarray(var.X, dtype=float)
    except Exception:
        return np.full(dim, 0.0)

def M_SMC(problem_type, params, data_info, hyperparameters, folder_path, is_hyper_optimize, is_m5=False, is_realETF=False, verbose=False):
    (kernel, nu, n_J, is_herding, T, time_limit) = (hyperparameters['kernel'], hyperparameters['nu'], hyperparameters['n_J'], hyperparameters['is_herding'], hyperparameters['T'], hyperparameters['time_Limit'])
    source_weights = hyperparameters.get('source_weights')
    if is_m5 or is_realETF:
        (M, N_i, n_sample, dim, X_all, data_emp, data_emp_all, N_test, X_test) = data_info
    else:
        (M, N_i, n_sample, M_test, N_test, dim, means_test, Sigma_test, X_all, data_emp, data_emp_all) = data_info
    (X_all_train, X_all_val) = ([], [])
    for i in range(M):
        (temp_train, temp_val) = train_test_split(X_all[i], test_size=0.2, random_state=42, shuffle=True)
        X_all_train.append(temp_train)
        X_all_val.append(temp_val)
    data_emp_train = np.vstack(X_all_train)
    data_emp_val = np.vstack(X_all_val)
    if kernel == 'Gaussian':
        (kernel_width, kernel_gamma) = median_heuristic(data_emp_train, data_emp_train)
        kernel_params = [kernel_width, kernel_gamma]
        if is_herding:
            X_all_herd_train = {}
            for i in range(M):
                X_all_herd_train[i] = kernel_herding(kernel, kernel_params, X_all_train[i], T)
            kernelMat = getKernel(X_all_herd_train, kernel_gamma)
        else:
            X_all_herd_train = X_all_train
            kernelMat = getKernel(X_all_herd_train, kernel_gamma)
    if kernel == 'Laplace':
        pass
    (kernel, nu, n_J, is_herding, T, time_limit) = (hyperparameters['kernel'], hyperparameters['nu'], hyperparameters['n_J'], hyperparameters['is_herding'], hyperparameters['T'], hyperparameters['time_Limit'])
    start_time = time.time()
    if kernel == 'Gaussian':
        (kernel_width, kernel_gamma) = median_heuristic(data_emp_all, data_emp_all)
        kernel_params = [kernel_width, kernel_gamma]
        if is_herding:
            X_all_herd = {}
            for i in range(M):
                X_all_herd[i] = kernel_herding(kernel, kernel_params, X_all[i], T)
            kernelMat = getKernel(X_all_herd, kernel_gamma)
        else:
            X_all_herd = X_all
            kernelMat = getKernel(X_all_herd, kernel_gamma)
    if kernel == 'Laplace':
        pass
    SMM_params = SMC(kernelMat, M, X_all, nu, folder_path, is_m5=False, is_realETF=False, is_show=True, source_weights=source_weights)
    SMC_firstpart_time = time.time() - start_time
    (w_SMC, obj_SMC, time_spent) = M_SMC_Model(problem_type, data_info, params, SMM_params, kernel, kernelMat, kernel_params, n_J, time_limit, X_all_herd, is_herding, is_m5, is_realETF, verbose)
    time_info = [SMC_firstpart_time + time_spent, SMC_firstpart_time, time_spent]
    return (w_SMC, obj_SMC, time_info)

def SMC(kernelMat, M, X_all, nu, folder_path='', is_m5=False, is_realETF=False, is_show=False, source_weights=None):
    if source_weights is None:
        upper_bounds = np.full(M, 1.0 / (M * nu), dtype=float)
    else:
        weights = np.asarray(source_weights, dtype=float).reshape(-1)
        if weights.shape[0] != M:
            raise ValueError(f'source_weights has length {weights.shape[0]} but expected {M}')
        if np.any(weights < 0):
            raise ValueError('source_weights must be nonnegative')
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError('source_weights must sum to a positive value')
        upper_bounds = weights / total / float(nu)
    (alpha, SVindex, BSVindex, outlierindex) = solvedual(kernelMat, float(np.max(upper_bounds)), upper_bounds=upper_bounds)
    SMC_info = [alpha, SVindex, BSVindex, outlierindex]
    if len(BSVindex) == 0:
        cap_ref = max(float(np.max(upper_bounds)), 1.0)
        indices = np.where(alpha > cap_ref * 0.001)[0]
        if len(indices) == 0:
            indices = np.where(alpha > 1e-12)[0]
        min_index = indices[np.argmin(alpha[indices])]
        R_square = kernelMat[min_index, min_index] - 2 * np.sum((alpha[i] * kernelMat[min_index, i] for i in range(M))) + np.sum((alpha[i] * alpha[j] * kernelMat[i, j] for i in range(M) for j in range(M)))
    else:
        alpha_BSM = alpha[BSVindex]
        BSMidx_R = np.argmin(alpha_BSM)
        R_square = kernelMat[BSVindex[BSMidx_R], BSVindex[BSMidx_R]] - 2 * np.sum((alpha[i] * kernelMat[BSVindex[BSMidx_R], i] for i in range(M))) + np.sum((alpha[i] * alpha[j] * kernelMat[i, j] for i in range(M) for j in range(M)))
    R_square = max(1e-08, R_square)
    R = math.sqrt(R_square)
    SMM_params = [alpha, R]
    return SMM_params

def M_SMC_Model(problem_type, data_info, params, SMM_params, kernel, K, kernel_params, n_J, time_limit, X_all_herd, is_herding=False, is_m5=False, is_realETF=False, verbose=False):
    if is_m5 or is_realETF:
        (M, N_i, n_sample, dim, X_all, data_emp, data_emp_all, N_test, X_test) = data_info
    else:
        (M, N_i, n_sample, M_test, N_test, dim, means_test, Sigma_test, X_all, data_emp, data_emp_all) = data_info
    if is_herding:
        data_emp_herd = {i: X_all_herd[i] for i in range(M)}
        N_i_herd = []
        data_emp_all_herd = []
        for i in range(M):
            N_i_herd.append(np.shape(data_emp_herd[i])[0])
            data_emp_all_herd.append(data_emp_herd[i])
        data_emp_all_herd = np.vstack(data_emp_all_herd)
        N_i_herd = np.array(N_i_herd)
        n_sample_herd = np.sum(N_i_herd)
    (alpha, R) = SMM_params
    start_time = time.time()
    if problem_type == 'newsvendor':
        (H, B, C, P, dim) = params
        A_X = np.array([C - P - B, C + H])
        A_Xi = np.array([B, -H - P])
        dim_x = dim
        n_k = 2
        obj = -1
        w_value = np.full(dim_x, 0.0)
        time_spent = 0.0
        model = None
        import copy
        samples_P_list_copy = copy.deepcopy(X_all)
        samples_P_list = {m: samples_P_list_copy[m] for m in range(M)}
        lhs_sampler = qmc.LatinHypercube(dim)
        lhs_samples = lhs_sampler.random(n_J)
        lb = np.min(data_emp_all, axis=0)
        ub = np.max(data_emp_all, axis=0)
        zetai = qmc.scale(lhs_samples, lb, ub)
        (alpha, R) = SMM_params
        if is_herding:
            data_emp_herd = {i: X_all_herd[i] for i in range(M)}
            N_i_herd = []
            data_emp_all_herd = []
            for i in range(M):
                N_i_herd.append(np.shape(data_emp_herd[i])[0])
                data_emp_all_herd.append(data_emp_herd[i])
            data_emp_all_herd = np.vstack(data_emp_all_herd)
            N_i_herd = np.array(N_i_herd)
            n_sample_herd = np.sum(N_i_herd)
        discretization_points = np.concatenate([data_emp_all, zetai])
        if is_herding:
            representation_points = np.concatenate([data_emp_all_herd, zetai])
        else:
            representation_points = np.concatenate([data_emp_all, zetai])
        if kernel == 'Gaussian':
            (kernel_width, kernel_gamma) = kernel_params
            g_fun = rkhsFun(rbf_kernel, gamma=kernel_gamma)
        K_emp_rep = rbf_kernel(data_emp_all, representation_points, gamma=kernel_gamma)
        K_rep_rep = rbf_kernel(representation_points, representation_points, gamma=kernel_gamma)
        K_disc_rep = rbf_kernel(discretization_points, representation_points, gamma=kernel_gamma)
        model = gp.Model('time_gap_generation')
        model.setParam('TimeLimit', time_limit)
        if not verbose:
            model.setParam(GRB.Param.OutputFlag, 0)
        w = model.addMVar(shape=dim_x, vtype=GRB.CONTINUOUS, lb=np.full(dim_x, 0), name='w')
        if is_herding:
            theta = model.addMVar(shape=n_J + n_sample_herd, lb=-10000, ub=10000, vtype=GRB.CONTINUOUS, name='theta')
        else:
            theta = model.addMVar(shape=n_J + n_sample, lb=-10000, ub=10000, vtype=GRB.CONTINUOUS, name='theta')
        beta = model.addVar(lb=-10000, ub=10000, vtype=GRB.CONTINUOUS, name='beta')
        temp = model.addVar(lb=-10000, ub=10000, vtype=GRB.CONTINUOUS, name='temp_max')
        t = model.addVar(lb=0, name='t')
        g_fun_obj_values = K_emp_rep @ theta
        temp_expr = 0
        for i in range(M):
            for j in range(N_i[i]):
                temp_expr = temp_expr + g_fun_obj_values[np.sum(N_i[:i]) + j] * alpha[i] / N_i[i]
        model.addConstr(temp >= temp_expr, name='empirical_expectation')
        import scipy.linalg
        jitter = 1e-06 * np.eye(K_rep_rep.shape[0])
        L = scipy.linalg.cholesky(K_rep_rep + jitter, lower=True)
        rkhs_norm = theta.T @ K_rep_rep @ theta
        model.addConstr(rkhs_norm <= t * t, name='rkhs_norm_bound')
        model.setParam(GRB.Param.QCPDual, 1)
        reg_term = R * t
        model.setObjective(beta + reg_term + temp, GRB.MINIMIZE)
        max_iterations = 100
        tolerance = 0.0001
        cuts_per_iter = 10
        max_stagnant_iters = 3
        stagnant_count = 0
        prev_obj = -10000000000.0
        obj_tolerance = 0.0001
        iteration = 0
        is_global_optimal = False
        start_time = time.time()
        try:
            print('Adding seed constraints...')
            num_seed_points = 3
            seed_indices = np.random.choice(len(discretization_points), num_seed_points, replace=False)
            for idx in seed_indices:
                zeta_seed = discretization_points[idx]
                g_expr_seed = K_disc_rep[idx, :] @ theta
                for k in range(n_k):
                    model.addConstr(A_X[k] @ w + A_Xi[k] @ zeta_seed <= beta + g_expr_seed, name=f'seed_{idx}_{k}')
            while iteration < max_iterations:
                iteration += 1
                print(f'--- Iteration {iteration} Start ---')
                model.optimize()
                status = model.Status
                if status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                    print(f'Main problem solving terminated abnormally, status code: {status}')
                    obj = model.ObjVal if model.SolCount > 0 else -1
                    w_value = _safe_mvar_value(w, dim_x)
                    break
                current_obj = model.ObjVal
                if abs(current_obj - prev_obj) < obj_tolerance:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                if stagnant_count >= max_stagnant_iters:
                    print(f'Iteration {iteration}: The objective function has improved minimally for {max_stagnant_iters} consecutive iterations (< {obj_tolerance}). Convergence assumed, triggering early termination!')
                    is_global_optimal = True
                    obj = current_obj
                    w_value = np.array([model.getVarByName(f'w[{l}]').X for l in range(dim_x)])
                    beta_value = beta.X
                    break
                prev_obj = current_obj
                w_val = w.X
                theta_val = theta.X
                beta_val = beta.X
                numeric_gvals = K_disc_rep @ theta_val
                violations = []
                for k in range(n_k):
                    if A_Xi[k].ndim == 1:
                        A_Xi_zeta = np.dot(discretization_points, A_Xi[k])
                    else:
                        A_Xi_zeta = discretization_points @ A_Xi[k].T
                    lhs_numeric = np.dot(A_X[k], w_val) + A_Xi_zeta
                    rhs_numeric = beta_val + numeric_gvals
                    viol_array = lhs_numeric - rhs_numeric
                    viol_indices = np.where(viol_array > tolerance)[0]
                    for idx in viol_indices:
                        violations.append({'k': k, 'idx': idx, 'viol': viol_array[idx]})
                if len(violations) == 0:
                    print(f'Iteration {iteration}: No violated constraints found, global optimum reached!')
                    is_global_optimal = True
                    obj = model.ObjVal
                    w_value = w.X
                    break
                violations.sort(key=lambda x: x['viol'], reverse=True)
                top_violations = violations[:cuts_per_iter]
                added_count = 0
                for v in top_violations:
                    k = v['k']
                    idx = v['idx']
                    zeta_v = discretization_points[idx]
                    g_expr = K_disc_rep[idx, :] @ theta
                    model.addConstr(A_X[k] @ w + A_Xi[k] @ zeta_v <= beta + g_expr)
                    added_count += 1
                print(f'Iteration {iteration}: Found {len(violations)} violated constraints, added Top {added_count} cuts. Current Obj: {model.ObjVal:.4f}')
                if time.time() - start_time > time_limit:
                    print('Reached total time limit.')
                    obj = model.ObjVal if model.SolCount > 0 else -1
                    w_value = _safe_mvar_value(w, dim_x)
                    break
            w_value = _safe_mvar_value(w, dim_x)
        except gp.GurobiError as e:
            print(f'Gurobi Error: {e}')
            obj = -2
            w_value = _safe_mvar_value(w, dim_x)
        except Exception as e:
            print(f'Other exception: {e}')
            obj = -2
            w_value = _safe_mvar_value(w, dim_x)
        finally:
            time_spent = time.time() - start_time
            print('Executing memory cleanup.')
            if model is not None:
                model.dispose()
    return (w_value, obj, time_spent)

def solve_rkhs(samples_list: Sequence[np.ndarray], validation_samples: np.ndarray, cost_params: Dict[str, np.ndarray], outdir: str, verbose: bool, seed: int, rkhs_nu: float, rkhs_source_weights: Sequence[float], rkhs_source_weight_mode: str, rkhs_n_J: int, rkhs_is_herding: bool, rkhs_T: int, rkhs_time_limit: int, gurobi_license_file: str='') -> Tuple[str, Optional[float], Optional[np.ndarray], Dict[str, Any]]:
    _ensure_gurobi_license(gurobi_license_file=gurobi_license_file)
    check_rkhs_dependencies()
    X_all = [np.asarray(s, dtype=float) for s in samples_list]
    M = len(X_all)
    N_i = np.array([s.shape[0] for s in X_all], dtype=int)
    data_emp = {i: X_all[i] for i in range(M)}
    data_emp_all = np.vstack(X_all)
    data_info = [M, N_i, int(N_i.sum()), int(data_emp_all.shape[1]), X_all, data_emp, data_emp_all, int(validation_samples.shape[0]), np.asarray(validation_samples, dtype=float)]
    params = [np.asarray(cost_params['H'], dtype=float), np.asarray(cost_params['B'], dtype=float), np.asarray(cost_params['C'], dtype=float), np.asarray(cost_params['P'], dtype=float), int(cost_params['dim'])]
    hyper = {'kernel': 'Gaussian', 'nu': float(rkhs_nu), 'source_weights': [float(v) for v in np.asarray(rkhs_source_weights, dtype=float).reshape(-1).tolist()], 'source_weight_mode': str(rkhs_source_weight_mode), 'n_J': int(rkhs_n_J), 'is_herding': bool(rkhs_is_herding), 'T': int(rkhs_T), 'time_Limit': int(rkhs_time_limit)}
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        if bool(verbose):
            (w, obj, time_info) = M_SMC('newsvendor', params, data_info, hyper, outdir, False, is_m5=False, is_realETF=True, verbose=True)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                (w, obj, time_info) = M_SMC('newsvendor', params, data_info, hyper, outdir, False, is_m5=False, is_realETF=True, verbose=False)
        return ('optimal' if w is not None else 'failed', None if obj is None else float(obj), None if w is None else np.asarray(w, dtype=float), {'roodso_hyper': hyper, 'nu': float(rkhs_nu), 'source_weight_mode': str(rkhs_source_weight_mode), 'source_weights': [float(v) for v in np.asarray(rkhs_source_weights, dtype=float).reshape(-1).tolist()]})
    except Exception as exc:
        return (f'error: {exc}', None, None, {'roodso_hyper': hyper, 'nu': float(rkhs_nu), 'source_weight_mode': str(rkhs_source_weight_mode), 'source_weights': [float(v) for v in np.asarray(rkhs_source_weights, dtype=float).reshape(-1).tolist()]})
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
