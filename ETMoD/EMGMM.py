import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from gridshift import GridShiftPP, preprocess_circular_linear_data_sc
from numpy.linalg import eigvals
from drawcluster_res import Draw_cluster_results
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import os

def wrap_to_pi(angle):
    """
    Wraps angles to the range [-pi, pi].

    Parameters:
        angle (float or numpy.ndarray): Input angle(s).

    Returns:
        float or numpy.ndarray: Angle(s) wrapped to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(angle): 
    """
    Wraps angles to the range [0, 2*pi].

    Parameters:
        angle (float or numpy.ndarray): Input angle(s).

    Returns:
        float or numpy.ndarray: Angle(s) wrapped to [0, 2*pi].
    """
    return (angle + 2 * np.pi) % (2 * np.pi)

def mle_complex(X):
    """
    MLE - Estimates the mean and covariance for given circular-linear data.
    
    Parameters:
        X (numpy.ndarray): Input data, where the first column should be circular 
                          and the second column should be linear.

    Returns:
        tuple: Mean vector (M) and covariance matrix (C).
    """
    if X.shape[1] == 2 and X.ndim == 2:
        # Circular components
        C = np.mean(np.cos(X[:, 1]))
        S = np.mean(np.sin(X[:, 1]))
        R = np.sqrt(C**2 + S**2)

        # Circular mean
        cr_m = np.arctan2(S, C)
        # -pi,pi

        # Linear mean
        l_m = np.mean(X[:, 0])
        M = np.array([l_m, cr_m])

        # Circular variance
        std = np.sqrt(-2 * np.log(R))
        V_c = std**2

        # Linear variance
        V_l = np.var(X[:, 0], ddof=1)

        # Covariance computation
        c = 0
        for j in range(X.shape[0]):
            c += (X[j, 0] - l_m) * wrap_to_pi(X[j, 1] - cr_m)
        c /= (X.shape[0] - 1)
        c = np.nan_to_num(c, nan=0.0)
        # cov between linear and circular

        # Covariance matrix
        C = np.array([[V_l, c], [c, V_c]])

        return M, C
    else:
        raise ValueError("The arguments should be 2-dimensional")

def ensure_positive_definite(matrix, epsilon=1e-6):
    """Ensures a covariance matrix is positive definite by adding epsilon to its diagonal."""
    try:
        if np.any(eigvals(matrix) <= 0):  # Check if it's not positive definite
            matrix += np.eye(matrix.shape[0]) * epsilon
    except LinAlgError:
        matrix = np.eye(matrix.shape[0]) * epsilon  # Fallback to identity matrix
    return matrix

def semi_wrapped_pdf(X, mu, Sigma, max_k=2):
    """Semi-wrapped probability density function."""
    linear, circular = X
    pdf_sum = 0
    for k in range(-max_k, max_k + 1):  # Consider a finite range of winding numbers
        shifted_theta = circular + 2 * np.pi * k 
        shifted_X = np.array([linear, shifted_theta])
        pdf_sum += multivariate_normal.pdf(shifted_X, mean=mu, cov=Sigma)
    return pdf_sum

# EM Algorithm Functions
def e_step_new(data, weights, means, covariances, max_k=3):
    """
    E-step: Compute responsibilities η_ijk for all data points, clusters, and wrapping values.
    """
    N, D = data.shape
    M = len(weights)  # Number of clusters
    responsibilities = np.zeros((N, M, 2 * max_k + 1))  # Responsibility matrix (i, j, k)

    for i in range(N):
        normalization_factor = 0.0
        for j in range(M):
            for k in range(-max_k, max_k + 1):
                shift = np.array([0, 2 * np.pi * k])
                prob = weights[j] * multivariate_normal.pdf(data[i], mean=means[j,:] + shift, cov=covariances[j,:,:])
                responsibilities[i, j, k + max_k] = prob  # Store for k-th wrapping
                normalization_factor += prob
            
            # Normalize responsibilities for (i, j, k)
        normalization_factor = max(normalization_factor, np.finfo(float).eps)  
        responsibilities[i, :, :] /= normalization_factor
    responsibilities[responsibilities < np.finfo(float).eps] = 0

    responsibilities = np.nan_to_num(responsibilities, nan=0.0, posinf=1e-6, neginf=1e-6)

           

    return responsibilities

def m_step_new(data, responsibilities, max_k=3):
    """
    M-step: Update parameters using responsibilities η_ijk.
    """
    N, D = data.shape
    M = responsibilities.shape[1]  # Number of clusters
    epsilon = 1e-4

    weights = np.zeros(M)
    means = np.zeros((M, D))
    covariances = np.zeros((M, D, D))

    for j in range(M):
        responsibility_sum = 0.0
        weighted_sum = np.zeros(D)
        weighted_cov_sum = np.zeros((D, D))

        for i in range(N):
            for k in range(-max_k, max_k + 1):
                eta_ijk = responsibilities[i, j, k + max_k]
                shift = np.array([0, 2 * np.pi * k])
                adjusted_data = data[i] - shift
                # adjusted_data[1] = wrap_to_pi(adjusted_data[1])

                responsibility_sum += eta_ijk
                weighted_sum += eta_ijk * adjusted_data
                weighted_cov_sum += eta_ijk * np.outer(adjusted_data, adjusted_data)
        
        responsibility_sum = max(responsibility_sum, np.finfo(float).eps)
        weights[j] = responsibility_sum / N
        means[j,:] = weighted_sum / responsibility_sum
        covariances[j,:,:] = weighted_cov_sum / responsibility_sum - np.outer(means[j,:], means[j,:])
        # make sure again
        covariances[j,:,:] = np.nan_to_num(covariances[j,:,:], nan=0.0, posinf=1e6, neginf=-1e6)
        covariances[j,:,:] += epsilon * np.eye(D)

    return weights, means, covariances

def log_likelihood_new(data, weights, means, covariances, max_k=3):
    """
    Compute the log-likelihood of the data given the current parameters.
    """
    N, D = data.shape
    M = len(weights)

    log_likelihood = 0.0
    for i in range(N):
        total_prob = 0.0
        for j in range(M):
            for k in range(-max_k, max_k + 1):
                shift = np.array([0, 2 * np.pi * k])
                total_prob += weights[j] * multivariate_normal.pdf(data[i], mean=means[j] + shift, cov=covariances[j])
                # total_prob += weights[j] * multivariate_normal.pdf(data[i], mean=means[j,:] + shift, cov=covariances[j,:,:])
        log_likelihood += np.log(total_prob)
    
    return log_likelihood

def swgmm_em_new(data, weights, means, covs, max_k=3, max_iter=100, tol=1e-3):
    """
    Semi-Wrapped Gaussian Mixture Model using the EM algorithm.
    """
    N, D = data.shape

    weights = weights
    means = means
    covariances = covs

    prev_log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-Step
        
        responsibilities = e_step_new(data, weights, means, covariances, max_k)

        # M-Step
        weights, means, covariances = m_step_new(data, responsibilities, max_k)

        # Compute log-likelihood
        
        current_log_likelihood = log_likelihood_new(data, weights, means, covariances, max_k)

        # Check for convergence
        if abs(current_log_likelihood - prev_log_likelihood) < tol:
            print(f"Converged at iteration {iteration}")
            break
        prev_log_likelihood = current_log_likelihood

        print(f"Iteration {iteration}: Log-Likelihood = {current_log_likelihood}")

    return weights, means, covariances
if __name__ == "__main__":
    data_dir = "/u/23/shij4/unix/Desktop/Gridcluster/SDE_data"
    param_dir = "/u/23/shij4/unix/Desktop/Gridcluster/EM_params"  
    output_dir = "/u/23/shij4/unix/Desktop/Gridcluster/SDE_params"  
    os.makedirs(output_dir, exist_ok=True)
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    param_files = sorted([f for f in os.listdir(param_dir) if f.endswith(".csv")])

    if len(data_files) != len(param_files):
        print("Warning! The file numbers are not match in SDE_data and EM_params！")
    all_data = []  # velocity
    all_means = []  # mean 
    all_covs = []  # cov 
    all_weights = []  # initial weight

    # loop all files
    results = []
    for data_file, param_file in zip(data_files, param_files):
        data_path = os.path.join(data_dir, data_file)
        param_path = os.path.join(param_dir, param_file)
        output_file = os.path.join(output_dir, f"sde_params_{data_file}")
        df_data = pd.read_csv(data_path)
        velocity_data = df_data[["speed_x", "speed_y"]].to_numpy()
        # all_data.append(velocity_data)

        # read EM parameters
        df_params = pd.read_csv(param_path)
        
        means = df_params[["mean_x", "mean_y"]].to_numpy()
        covs = df_params[["cov_00", "cov_01", "cov_10", "cov_11"]].to_numpy()
        cov_matrices = [covs[i].reshape(2, 2) for i in range(len(covs))]
        cov_matrices = np.array(cov_matrices)
        
        
        max_cluster = df_params["cluster_id"].max()
        num_clusters = max_cluster + 1  
        weights = np.ones(num_clusters) / num_clusters  
        for j in range(max_cluster+1):
            f = np.diag(cov_matrices[j,:,:])  # Extract the diagonal elements of the covariance matrix
            f = np.floor(np.log10(f))  # Take the base-10 logarithm and floor it
            cov_matrices[j,:,:] = np.diag([10**(f[0] - 1), 10**(f[1] - 1)])  # Set the diagonal elements to be adjusted by a power of 10
            epsilon = 1e-6  # small regularization constant
            cov_matrices[j,:,:] += epsilon * np.eye(2)

        weights_new, means_new, covs_new = swgmm_em_new(velocity_data, weights, means, cov_matrices, max_k=2)
        cov_flattened = [covs_new[i].flatten() for i in range(num_clusters)]
        
        df_result = pd.DataFrame({
        "mean_x": means_new[:, 0],
        "mean_y": means_new[:, 1],
        "cov_00": [cov[0] for cov in cov_flattened],
        "cov_01": [cov[1] for cov in cov_flattened],
        "cov_10": [cov[2] for cov in cov_flattened],
        "cov_11": [cov[3] for cov in cov_flattened],
        "weight": weights_new,
        })

        df_result.to_csv(output_file, index=False)
        print(f"Saved file: {output_file}")