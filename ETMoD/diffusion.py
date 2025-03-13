import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gridshift import preprocess_circular_linear_data_sc, GridShiftPP
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import DataLoader, TensorDataset
from EMGMM import swgmm_em_new, mle_complex, wrap_to_pi, semi_wrapped_pdf

# Set random seeds for reproducibility
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(1)

# Select device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
file_path_1 = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/output/grid_0_interval_1351040400.csv"
file_path_2 = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/output/grid_0_interval_1351042200.csv"
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

df1_half = len(df1) // 2
df2_half = len(df2) // 2

df1_second_half = df1.iloc[df1_half:]
df2_first_half = df2.iloc[:df2_half]
GS = GridShiftPP(bandwidth=1, iterations=300)

# Combine datasets
new_train_data = pd.concat([df1_second_half, df2_first_half], ignore_index=True)
velocity = new_train_data.iloc[:, 4].to_numpy()
motion_angle = new_train_data.iloc[:, 5].to_numpy()
circular_linear_data = np.column_stack((velocity, motion_angle))
trans_data = preprocess_circular_linear_data_sc(circular_linear_data, w=3)
centers, membership = GS.fit_predict(trans_data)
unique_clusters = np.unique(membership)
# print("centers", centers)

means = []
cluster_data = []
for cluster in unique_clusters:
    # print("1")
    cluster_data.append(circular_linear_data[membership == cluster])
    mean,covs = mle_complex(circular_linear_data[membership == cluster])
    means.append(mean)
# print(means)

# Define dataset
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)  # Move to GPU

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].view(2, 1, 1)  # Reshape to (2, 1, 1)
        img = img.expand(-1, 8, 8)  # Expand to (2, 8, 8)
        return img

dataset = TabularDataset(cluster_data[1])
# Define the UNet model
model = Unet(
    dim=64,  # Dimension of the feature maps
    dim_mults=(1, 2, 4, 8),  # Multipliers for the dimensions
    channels=2  # Treat tabular data as 2-channel "images"
).to(device)

# Define the diffusion model
diffusion = GaussianDiffusion(
    model,
    image_size=8,  # Change from 1 to 8
    timesteps=1000
).to(device)
batch_size = 128
num_epochs = 100
learning_rate = 1e-4

# Create DataLoader for batching
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer (AdamW is good for diffusion models)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch in dataloader:
        batch = batch.to(device)  # Move batch to GPU if available

        optimizer.zero_grad()  # Reset gradients
        # print(batch.shape)

        loss = diffusion(batch)  # Compute loss using diffusion model

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)  # Compute average loss for epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# # Save model after training
# torch.save(model.state_dict(), "diffusion_model.pth")
# print("Model saved successfully!")
# Load the trained model
model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully!")
num_samples = 1000  # Number of samples to generate

# # Generate noise input (random Gaussian noise)
samples = diffusion.sample(batch_size = num_samples).detach().cpu().numpy()  # Convert to NumPy
# # print(samples.shape)
# # Convert 8x8 samples back to 2D (velocity, motion_angle)
original_shape_samples = samples.mean(axis=(2, 3))  # Average over (8,8) pixels
# print(original_shape_samples.shape)
generated_df = pd.DataFrame(original_shape_samples, columns=["velocity", "motion_angle"])
generated_df.to_csv("generated_samples.csv", index=False)
print("Generated samples saved to generated_samples.csv")
file_path_3 = "/u/23/shij4/unix/Desktop/Gridcluster/generated_samples.csv"
df3 = pd.read_csv(file_path_3).to_numpy()
transformed_data = preprocess_circular_linear_data_sc(df3, w=3)
centers, membership = GS.fit_predict(transformed_data)
# print(centers)
means = []
cluster_data = []
covs = []
for cluster in unique_clusters:
    cluster_data.append(df3[membership == cluster])
    mean,cov = mle_complex(df3[membership == cluster])
    means.append(mean)
    covs.append(cov)
print(means)
