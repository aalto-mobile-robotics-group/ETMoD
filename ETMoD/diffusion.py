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
file_path_1 = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/output/grid_*_interval_****.csv"
file_path_2 = "/u/23/shij4/unix/Desktop/Gridcluster/grid_1024_drop/output/grid_*_interval_****.csv"
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

means = []
cluster_data = []
for cluster in unique_clusters:
    cluster_data.append(circular_linear_data[membership == cluster])
    mean,covs = mle_complex(circular_linear_data[membership == cluster])
    means.append(mean)

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

model = Unet(
    dim=64,  
    dim_mults=(1, 2, 4, 8),  
    channels=2  
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=8,  
    timesteps=1000
).to(device)
batch_size = 128
num_epochs = 100
learning_rate = 1e-4

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch in dataloader:
        batch = batch.to(device)  

        optimizer.zero_grad()  

        loss = diffusion(batch)  

        loss.backward() 
        optimizer.step()  

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# # Save model after training
torch.save(model.state_dict(), "diffusion_model.pth")
# Load the trained model
model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully!")
num_samples = 1000  # Number of samples to generate

samples = diffusion.sample(batch_size = num_samples).detach().cpu().numpy()  # Convert to NumPy
original_shape_samples = samples.mean(axis=(2, 3))  
generated_df = pd.DataFrame(original_shape_samples, columns=["velocity", "motion_angle"])
generated_df.to_csv("generated_samples.csv", index=False)

