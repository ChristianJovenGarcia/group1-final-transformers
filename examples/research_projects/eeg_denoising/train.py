from transformers import EEGDenoisingConfig, EEGDenoisingModel
from utils import TUARDataset
from torch.utils.data import DataLoader, RandomSampler
import torch

# Initialize the dataset
# Initialize the dataset
dataset = TUARDataset(fixed_length=config.context_length + config.prediction_length)

# Use a sampler with the correct number of samples
sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))

# Create the data loader
data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Initialize the model configuration
config = EEGDenoisingConfig(
    num_channels=64,
    input_size=64,
    window_size=256,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=8,
    scaling="mean",
)

# Initialize the model
model = EEGDenoisingModel(config)

# Train
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    for batch in data_loader:
        optimizer.zero_grad()

        # Unpack the batch
        eeg_data, past_time_features, past_observed_mask = batch

        # Ensure past_observed_mask matches the shape of eeg_data
        if past_observed_mask.ndim == 2:
            past_observed_mask = past_observed_mask.unsqueeze(-1)

            # Forward pass
            denoised = model(eeg_data, past_time_features, past_observed_mask)

        if 'clean_eeg' in locals() or 'clean_eeg' in globals():
            # Use clean EEG data as the target if available
            loss = torch.nn.functional.mse_loss(denoised, clean_eeg)
        else:
            # Fall back to using noisy EEG data as the target
            loss = torch.nn.functional.mse_loss(denoised, eeg_data)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item()}")