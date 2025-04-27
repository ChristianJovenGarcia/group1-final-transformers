from transformers import EEGDenoisingConfig, EEGDenoisingModel
from utils import TUARDataset
from torch.utils.data import DataLoader, RandomSampler
import torch

# Initialize the dataset
dataset = TUARDataset()

# Check if the dataset is empty
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check the dataset path or preprocessing steps.")

# Use a sampler with the correct number of samples
sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))

print(f"Dataset size: {len(dataset)}")

# Create the data loader
data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Initialize the model configuration
config = EEGDenoisingConfig(
    num_channels=64,
    window_size=256,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=8,
    scaling="mean",
    num_static_categorical_features=0,  # Ensure this is set
)

# Initialize the model
model = EEGDenoisingModel(config)

# Train
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    for batch in data_loader:
        optimizer.zero_grad()
        denoised = model(batch)
        loss = torch.nn.functional.mse_loss(denoised, batch)  # Replace with clean data
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")