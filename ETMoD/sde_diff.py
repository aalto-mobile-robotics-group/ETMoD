from network import SDE
import torch
import os
import pandas as pd
import torchsde
import numpy as np

import torch.nn as nn
from torch.distributions import Normal, Independent
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import glob
from scipy.optimize import linear_sum_assignment  
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from KLD import compute_divergence, generate_semi_wrapped_samples, sample_semi_wrapped_normal
from EMGMM import wrap_to_2pi, wrap_to_pi, mle_complex, semi_wrapped_pdf
from gridshift import preprocess_circular_linear_data_sc, GridShiftPP

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

folder_path = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/em_**"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_all_seeds(1)
pattern = os.path.join(folder_path, "grid_**_interval_*_em_results.npz")

# Get all matching files
grid_0_files = sorted(glob.glob(pattern))

all_means_padded = []
all_covs_padded = []
all_weight_padded = []
timestamps_padded = []

prev_means = None
max_k = 8
def angular_distance(theta1, theta2):
    """Compute angular distance"""
    diff = np.abs(theta1 - theta2)  
    return np.minimum(diff, 2 * np.pi - diff)

def match_clusters(prev_means, curr_means):
    """Match clusters based on the minimum combined distance (speed and angle)."""
    if prev_means is None:
        return np.arange(len(curr_means)), np.arange(len(curr_means))  # Identity mapping for first step
    
    prev_speeds = prev_means[:, 0]
    prev_angles = prev_means[:, 1]
    curr_speeds = curr_means[:, 0]
    curr_angles = curr_means[:, 1]

    n_prev = len(prev_speeds)
    n_curr = len(curr_speeds)
    speed_cost = np.zeros((n_prev, n_curr))  
    angle_cost = np.zeros((n_prev, n_curr))  

    for i in range(n_prev):
        for j in range(n_curr):
            speed_cost[i, j] = np.abs(prev_speeds[i] - curr_speeds[j])

    for i in range(n_prev):
        for j in range(n_curr):
            angle_cost[i, j] = angular_distance(prev_angles[i], curr_angles[j])

    cost_matrix = speed_cost + angle_cost

    col_ind = np.argmin(cost_matrix, axis=1)  
    row_ind = np.arange(n_prev) 

    return np.array(row_ind), np.array(col_ind)

def pad_clusters(means, covs, weights, max_k):
    """Pad means, covs, and weights to max_k clusters."""
    k = means.shape[0]
    if k < max_k:
        pad_size = max_k - k
        means = np.vstack((means, np.zeros((pad_size, means.shape[1]))))
        covs = np.vstack((covs, np.tile(np.eye(covs.shape[1]), (pad_size, 1, 1))))
        # weights = np.hstack((weights, np.zeros(pad_size)))
    return means, covs
index = []

for t, param_file in enumerate(grid_0_files):
    
    means = np.load(param_file)["means"]
    if means.shape[0] == 0:
        continue
    covs = np.load(param_file)["covs"]
    weights = np.load(param_file)["weights"]

    if prev_means is not None:

        row_ind, col_ind = match_clusters(prev_means, means)
        index.append(col_ind)
 
    prev_means = means
    means, covs = pad_clusters(means, covs, weights, max_k)
    all_means_padded.append(means)
    all_covs_padded.append(covs)
    all_weight_padded.append(weights)
    
all_means_padded = torch.tensor(np.stack(all_means_padded), dtype=torch.float32).to(device)  # (N, max_k, 3)
all_covs_padded = torch.tensor(np.stack(all_covs_padded), dtype=torch.float32).to(device) # (N, max_k, 2, 2)
# all_means_padded[:,:,1] = wrap_to_2pi(all_means_padded[:,:,1])
num_epochs = 100
x_t = all_means_padded
min_loss = float("inf")
time_series_vel = []  # For velocity
time_series_angle = []  # For angle
time_series_cov = []
# Initial data point
current_index = 2  # Start from x_t[0, 0, :]
current_data_vel = x_t[0, current_index, 0].item()  # Velocity
current_data_angle = x_t[0, current_index, 1].item()  # Angle
current_data_cov = all_covs_padded[0, current_index, :].cpu().numpy()
time_series_vel.append(current_data_vel)
time_series_angle.append(current_data_angle)
time_series_cov.append(current_data_cov)

# Traverse through time steps
for t in range(len(index)):
    current_index = index[t][current_index]  # Update current_index
    current_data_vel = x_t[t + 1, current_index, 0].item()  # Velocity
    current_data_angle = x_t[t + 1, current_index, 1].item()  # Angle
    current_data_cov = all_covs_padded[t + 1, current_index, :].cpu().numpy()
    time_series_vel.append(current_data_vel)
    time_series_angle.append(current_data_angle)
    time_series_cov.append(current_data_cov)

# Combine into a single array
time_series = np.column_stack((time_series_vel, time_series_angle))  # Shape: (num_timesteps, 2)
time_series_cov = np.array(time_series_cov)


x = time_series
# Convert to Tensor
x_t = torch.tensor(x, device=device).reshape(-1, 1, 2).to(torch.float32)

ts = torch.linspace(0, 1, x_t.shape[0]).to(device).to(torch.float32)
state_size = x_t.shape[2]

sde = SDE(n_inputs = state_size, n_outputs = state_size, device = device)
optimizer = optim.Adam(sde.parameters(), lr=0.01)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()

    ys = torchsde.sdeint(sde, x_t[0], ts, method='milstein').to(device)

    loss = F.smooth_l1_loss(ys, x_t)  

    loss.backward()

    optimizer.step()

    epoch_loss += loss.item()
    if epoch_loss <= min_loss:
        min_loss = epoch_loss
        torch.save(sde.state_dict(), "/u/23/shij4/unix/Desktop/Gridcluster/sde_model_grid**.pth")
        print(f"Save model, minimum Loss: {epoch_loss:.4f}")
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
torch.save(sde.state_dict(), "/u/23/shij4/unix/Desktop/Gridcluster/sde_model_grid**.pth")
sde.load_state_dict(torch.load("/u/23/shij4/unix/Desktop/Gridcluster/sde_model_grid**.pth"))
sde.eval()
with torch.no_grad():
    ys = torchsde.sdeint(sde, x_t[0], ts, method='milstein') 
ys_np = ys.cpu().numpy().squeeze()  


from universal_divergence import estimate
gt_data_path = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/output/grid_**_interval_*.csv"

file_paths = glob.glob(gt_data_path)

vel_target = "***"
angle_target = "***"
vel_tolerance = "***"
angle_tolerance = "***"
# adjust according to data

for t, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    
    velocity = df.iloc[:, 4].to_numpy()
    motion_angle = df.iloc[:, 5].to_numpy()
    circular_linear_data = np.column_stack((velocity, motion_angle))
    circular_linear_data[:,1] = wrap_to_2pi(circular_linear_data[:,1])

    filtered_data = circular_linear_data[
        (np.abs(circular_linear_data[:, 0] - vel_target) < vel_tolerance) & 
        (np.abs(circular_linear_data[:, 1] - angle_target) < angle_tolerance)
    ]
    sde_samples = sample_semi_wrapped_normal(ys_np[t], time_series_cov[t], n_samples=filtered_data.shape[0])
    divergence = estimate(filtered_data, sde_samples, k=2)

    print(divergence)
