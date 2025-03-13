import os
import csv
import re
from datetime import datetime
from gridshift import preprocess_circular_linear_data_sc, GridShiftPP
from EMGMM import swgmm_em_new, mle_complex, wrap_to_pi, semi_wrapped_pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_file(file_path):
    df = pd.read_csv(file_path, header=0)
    velocity = df.iloc[:, 4].to_numpy()  # Fifth column
    motion_angle = df.iloc[:, 5].to_numpy()  # Sixth column
    circular_linear_data = np.column_stack((velocity, motion_angle))
    
    transformed_data = preprocess_circular_linear_data_sc(circular_linear_data, w=3)
    model = GridShiftPP(bandwidth=1, iterations=300)
    centers, membership = model.fit_predict(transformed_data)
    membership = np.array(membership)
    
    unique_clusters = np.unique(membership)
    means = []
    covs = []
    
    for cluster in unique_clusters:
        cluster_data = circular_linear_data[membership == cluster]
        
        mean, cov = mle_complex(cluster_data)
        means.append(mean)
        covs.append(cov)
    
    weights = np.ones(centers.shape[0]) / centers.shape[0]
    means = np.array(means, dtype=np.float16)
    covs = np.array(covs, dtype=np.float16)
    
    for j in range(centers.shape[0]):
        f = np.diag(covs[j, :, :])  # Extract diagonal elements
        f = np.floor(np.log10(f))  # Take base-10 log and floor it
        covs[j, :, :] = np.diag([10 ** (f[0] - 1), 10 ** (f[1] - 1)])
        epsilon = 1e-6  # Small regularization constant
        covs[j, :, :] += epsilon * np.eye(2)
    
    weights_new, means_new, covs_new = swgmm_em_new(circular_linear_data, weights, means, covs, max_k=2)
    means_new[:, 1] = wrap_to_pi(means_new[:, 1])
    linear_range = np.linspace(0, 6, 100)  
    circular_range = np.linspace(-np.pi, np.pi, 100)
    linear_grid, circular_grid = np.meshgrid(linear_range, circular_range)
    fig, ax = plt.subplots(figsize=(8, 6))

    for j in range(len(weights_new)):
        pdf_values = np.zeros_like(linear_grid)
        for i in range(linear_grid.shape[0]):
            for k in range(linear_grid.shape[1]):
                X = [linear_grid[i, k], circular_grid[i, k]]
                pdf_values[i, k] = semi_wrapped_pdf(X, means_new[j], covs_new[j])

        ax.contour(circular_grid, linear_grid, pdf_values, levels=15, cmap='viridis')
    data = circular_linear_data
    linear_data = data[:, 0]  
    circular_data = data[:, 1] 
    ax.scatter(circular_data, linear_data, c='blue', s=10, label='Data')

    for j, mean in enumerate(means_new):
        ax.scatter(mean[1], mean[0], c='red', marker='x', s=100, label=f'Cluster {j+1} Mean')

    plt.xlabel('Orientation [rad]')
    plt.ylabel('Amplitude [m/s]')
    plt.title('EM Results for SWND')
    plt.legend()
    plt.grid(True)
    plt.savefig("EMSWND_test")
    print("one fig saved")
    
    return weights_new, means_new, covs_new

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    processed_files = {f.replace("_em_results.npz", ".csv") for f in os.listdir(output_folder) if f.endswith(".npz")}
    print("Porcessed files", processed_files)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv") and file_name not in processed_files:
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_name}")
            
            weights, means, covs = process_file(file_path)
            
            output_file = os.path.join(output_folder, file_name.replace(".csv", "_em_results.npz"))
            np.savez(output_file, weights=weights, means=means, covs=covs)
            print(f"Saved results to {output_file}")

if __name__ == "__main__":
    input_folder = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/output"
    output_folder = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/em_results"
    main(input_folder, output_folder)