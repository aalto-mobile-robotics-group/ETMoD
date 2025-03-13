import numpy as np
from collections import defaultdict, Counter
from sklearn.datasets import make_blobs
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from drawcluster_loc import Draw_cluster_loc
from drawcluster_res import Draw_cluster_results
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_circular_linear_data_sc(data, w):
    """
    Preprocesses circular-linear data to handle angular variables.

    Parameters
    ----------
    data : ndarray of shape (n_samples, 2)
        The input data where the first column is velocity (linear)
        and the second column is motion_angle (angular in radians).

    Returns
    -------
    transformed_data : ndarray
        The transformed data with angular variable replaced by its sin and cos components.
    """
    velocity = data[:, 0]  # Linear data (velocity)
    angles = data[:, 1]    # Angular data (motion_angle)
    
    # Transform angular data to sin and cos
    sin_component = np.sin(angles)
    cos_component = np.cos(angles)
    
    # Concatenate velocity with sin and cos components of angles
    transformed_data = np.column_stack((velocity, w * cos_component, w * sin_component))
    # transformed_data = np.column_stack(velocity, theta)
    return transformed_data

def check_convergence(cluster_grid, new_cluster_grid):
    # Check if the keys are the same
    if cluster_grid.keys() != new_cluster_grid.keys():
        return False

    # Check if the corresponding values (the [sum, count] arrays) are the same
    for key in cluster_grid:
        old_sum, old_count = cluster_grid[key]
        new_sum, new_count = new_cluster_grid[key]
        
        # Use np.allclose for sum comparison (for floating-point values)
        if not np.allclose(old_sum, new_sum) or old_count != new_count:
            return False

    return True

class GridShiftPP:
    def __init__(self, bandwidth, iterations=None):
        """
        Parameters
        ----------
        
        bandwidth: Radius for binning points. Points are assigned to the bin 
                corresponding to floor division by bandwidth

        iterations: Maximum number of iterations to run.

        """
        self.bandwidth = bandwidth
        self.iterations = iterations

    def generate_offsets(self, d, base):
        """
        Generate 3**d neighbors for any point.

        Parameters
        ----------
        d: Dimensions
        base: 3, corresponding to (-1, 0, 1)
        offsets: (3**d, d) array of offsets to be added to 
                 a bin to get neighbors
        """
        offsets = np.full((base ** d, d), -1, dtype=np.int32)
        for i in range(base ** d):
            tmp = i
            
            for j in range(d):
                if tmp == 0:
                    break
                offsets[i, j] = tmp % base - 1
                tmp //= base
        return offsets

    def fit_predict(self, X):
        """
        Each shift has two steps: First, points are binned based on floor 
        division by bandwidth. Second, each bin is shifted to the 
        weighted mean of its 3**d neighbors. 
        Lastly, points that are in the same bin are clustered together.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
           Euclidean space

        Returns
        ----------
        (n, ) cluster labels
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        X_shifted = np.copy(X)
        membership = np.full(n, -1, dtype=np.int32)
        # membership_new = np.full(n, -1, dtype=np.int32)

        iteration = 0
        base = 3
        offsets = self.generate_offsets(d, base)

        cluster_grid = defaultdict(lambda: [np.zeros(d), 0])
        map_cluster = {}
        temp = 0
        

        # Initial clustering into bins
        for i in range(n):
            bin_key = tuple((X_shifted[i] // self.bandwidth).astype(int))
            if bin_key not in map_cluster:
                map_cluster[bin_key] = temp
                temp += 1
            cluster_grid[bin_key][0] += X_shifted[i]
            cluster_grid[bin_key][1] += 1
            membership[i] = map_cluster[bin_key]
        # print("cluster_grid_old",cluster_grid)

        # Iterative process
        for _ in range(self.iterations):
            means = defaultdict(lambda: [np.zeros(d), 0])
            temp_new = 0
            for bin_key, (bin_sum, bin_count) in cluster_grid.items():
                for offset in offsets:
                    neighbor_bin = tuple(np.add(bin_key, offset))
                   
                    if neighbor_bin in cluster_grid:
                       
                        if neighbor_bin not in means:
                            means[neighbor_bin][0] = np.zeros(d)
                            means[neighbor_bin][1] = 0
                        means[neighbor_bin][0] += bin_sum
                        means[neighbor_bin][1] += bin_count
            
            for bin_coord in cluster_grid.keys():
                if bin_coord in means:
                    mean_sum, mean_count = means[bin_coord]
                    cluster_grid[bin_coord][0] = mean_sum / mean_count
                    # update mean
            # print(means)
            new_cluster_grid = defaultdict(lambda: [np.zeros(d), 0])
            new_map_cluster = {}

            for bin_coord, (mean, count) in cluster_grid.items():
                new_bin_coord = tuple((mean // self.bandwidth).astype(int))
                # print("new_bin_coord",new_bin_coord)
                new_cluster_grid[new_bin_coord][0] += mean*count
                new_cluster_grid[new_bin_coord][1] += count
                # if new_bin_coord not in new_map_cluster:
                #     new_map_cluster[new_bin_coord] = temp_new
                #     temp_new += 1
            # for new_bin_coord in new_map_cluster:
            #     for i in range(n):
            #         if 
            # print(cluster_grid,new_cluster_grid)
            if check_convergence(cluster_grid,new_cluster_grid):
                print("Converge.")
                break

            cluster_grid = new_cluster_grid

        # print("cluster_grid",cluster_grid)
        cluster_centers = np.array([sum_points / count for sum_points, count in cluster_grid.values()])
       
        for i in range(n):
            distances = [distance.euclidean(X[i], center) for center in cluster_centers]
            membership[i] = np.argmin(distances)
        # print("cluster_centers",cluster_centers)
        # print("membership",membership)
        return cluster_centers, membership
       
if __name__ == "__main__":

    # Example usage
    bandwidth = 1
    iterations = 300
    file_path = "/u/23/shij4/unix/Desktop/Gridcluster/test_data_time/1352853.csv"
    df = pd.read_csv(file_path, header=0)
    pos_x = df.iloc[:, 2].to_numpy()  # third column
    pos_y = df.iloc[:, 3].to_numpy()  # forth column
    velocity = df.iloc[:, 4].to_numpy()  # Fifth column
    motion_angle = df.iloc[:, 5].to_numpy()  # Sixth column
    w = 3 # for setting the polar parameter

    # Combine into a single array
    pos_data = np.column_stack((pos_x, pos_y))
    circular_linear_data = np.column_stack((velocity, motion_angle))
    transformed_data = preprocess_circular_linear_data_sc(circular_linear_data, w)
    pos_circular_linear_data = np.column_stack((pos_data, transformed_data))
    model = GridShiftPP(bandwidth=bandwidth, iterations=iterations)
    centers, membership = model.fit_predict(transformed_data)
    print("centers", centers)

    centers_pos, membership_pos = model.fit_predict(pos_data)
    print("centers", centers_pos)
    membership = membership.astype(int)
    membership_pos = membership_pos.astype(int)

    max_speed_clusters = membership.max() + 1

    # Combine the clusters into a single label
    combined_labels = membership_pos * max_speed_clusters + membership
    Draw_cluster_loc(df, membership_pos)
    # membership_df = pd.DataFrame({'membership_test_pos': membership_pos})
    # membership_df.to_csv('membership_test_pos.csv', index=False)