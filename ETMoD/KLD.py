import numpy as np
from scipy.spatial import KDTree
from EMGMM import swgmm_em_new, mle_complex, wrap_to_pi, semi_wrapped_pdf

def compute_divergence(V_o, V_s, k=5):
    """
    Computes the divergence estimator given two sets of observations.
    
    Parameters:
    - V_o: np.array of shape (n, d) -> Observations from true distribution
    - V_s: np.array of shape (m, d) -> Observations from estimated distribution
    - k: Number of nearest neighbors to consider
    
    Returns:
    - Estimated divergence value
    """
    n, d = V_o.shape
    m = V_s.shape[0]
    
    # Build KD-trees for efficient nearest neighbor search
    tree_o = KDTree(V_o)
    tree_s = KDTree(V_s)
    
    # Compute rho_k(i): distance to k-th nearest neighbor within V_o
    rho_k = np.array([tree_o.query(V_o[i], k+1)[0][-1] for i in range(n)])  # k+1 because first is itself
    
    # Compute nu_k(i): distance to k-th nearest neighbor within V_s
    nu_k = np.array([tree_s.query(V_o[i], k)[0][-1] for i in range(n)])  # No need for k+1 as V_o is different from V_s
    
    # Compute divergence estimator
    divergence = (d / n) * np.sum(np.log2(nu_k / rho_k)) + np.log2(m / (n - 1))
    divergence = max(0, divergence)
    return divergence

def generate_semi_wrapped_samples(mean, cov, n_samples=1000, max_k=2):
    """
    Generate random samples from a semi-wrapped distribution using rejection sampling.
    
    Parameters:
    - mean: Mean of the distribution (linear, circular).
    - cov: Covariance matrix of the distribution.
    - n_samples: Number of samples to generate.
    - max_k: Maximum winding number for the semi-wrapped PDF.
    
    Returns:
    - samples: Generated samples (n_samples, 2).
    """
    # Define the bounding box for rejection sampling
    linear_range = (mean[0] - np.sqrt(cov[0, 0]), mean[0] + np.sqrt(cov[0, 0]))  # Linear variable range
    circular_range = (mean[1] - np.sqrt(cov[1, 1]), mean[1] + np.sqrt(cov[1, 1]))  # Circular variable range
    
    samples = []
    while len(samples) < n_samples:
        # Generate a candidate sample
        linear_candidate = np.random.uniform(linear_range[0], linear_range[1])
        circular_candidate = np.random.uniform(circular_range[0], circular_range[1])
        X_candidate = np.array([linear_candidate, circular_candidate])
        
        # Compute the PDF value for the candidate
        pdf_value = semi_wrapped_pdf(X_candidate, mean, cov, max_k=max_k)
        
        # Accept or reject the candidate
        if np.random.uniform(0, 1) < pdf_value:
            samples.append(X_candidate)
    
    return np.array(samples)

def sample_semi_wrapped_normal(mean, cov, n_samples=1000, max_k=2):
    """
    Generate samples from a semi-wrapped normal distribution efficiently.

    Parameters:
    - mean: Mean vector [mu_x, mu_theta]
    - cov: Covariance matrix [[var_x, cov_xy], [cov_xy, var_theta]]
    - n_samples: Number of samples to generate
    - max_k: Maximum winding number for the semi-wrapped PDF

    Returns:
    - samples: Array of shape (n_samples, 2) with linear and wrapped circular samples
    """
    # Sample directly from the multivariate normal
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Wrap the circular component to [-pi, pi]
    samples[:, 1] = (samples[:, 1] + np.pi) % (2 * np.pi) - np.pi

    return samples


