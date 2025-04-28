from transformers.models.eeg_denoising.configuration_eeg_denoising import EEGDenoisingConfig
from utils import TUARDataset
from torch.utils.data import DataLoader, RandomSampler
from transformers.models.eeg_denoising.modeling_eeg_denoising import EEGDenoisingModel
import torch
import os
import numpy as np
import mne
from transformers import TimeSeriesTransformerModel, PreTrainedModel
from transformers import logging
logger = logging.get_logger(__name__)


# Define the configuration
config = EEGDenoisingConfig(
    context_length=128,  # Example value, adjust as needed
    prediction_length=32,  # Example value, adjust as needed
    num_channels=64,  # Example value, adjust as needed
    input_size=64,  # Example value, adjust as needed
    scaling="mean",  # Example value, adjust as needed
)

# Replace "/path/to/data" with the actual path to your dataset
dataset_path = "/workspaces/group1-final-transformers/examples/research_projects/eeg_denoising/data/physionet.org/files/eegmmidb/1.0.0/"

# Check if the dataset directory exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset directory does not exist: {dataset_path}")
    exit(1)

# Initialize the dataset with a maximum of 50 files
dataset = TUARDataset(data_dir=dataset_path, fixed_length=20000, max_files=50)

# Process the dataset
processed_count = 0
for idx in range(len(dataset)):
    eeg_data, past_time_features, past_observed_mask = dataset[idx]

    # Skip files that failed to process
    if eeg_data is None or past_time_features is None or past_observed_mask is None:
        print(f"Skipping file at index {idx} due to loading error.")
        continue

    # Pass the data to the model
    try:
        denoised = model(eeg_data, past_time_features, past_observed_mask)
        processed_count += 1

        # Stop processing after 40-50 files
        if processed_count >= 50:
            print("Processed the maximum number of files. Stopping.")
            break

    except Exception as e:
        print(f"Error during model processing for file {idx}: {e}")
        continue

# Use a sampler with the correct number of samples
sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))

# Create the data loader
data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Initialize the model configuration
config = EEGDenoisingConfig(
    context_length=128,
    prediction_length=32,
    num_channels=64,
    input_size=128,      # <--- CHANGE THIS TO 128
    feature_size=128,    # <--- CHANGE THIS TO 128
    hidden_size=128,     # <--- CHANGE THIS TO 128
    num_hidden_layers=4,
    num_attention_heads=8,
    scaling="mean",
    d_model=128          # <--- CHANGE THIS TO 128 if used
)

# Initialize the model
model = EEGDenoisingModel(config)

# Train
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    loss = None  # Initialize loss as None at the start of each epoch
    for batch in data_loader:
        optimizer.zero_grad()

        # Unpack the batch
        eeg_data, past_time_features, past_observed_mask = batch
        # Ensure dataset tensors are float32
        eeg_data = eeg_data.float()
        past_time_features = past_time_features.float()
        past_observed_mask = past_observed_mask.float()

        # Ensure past_observed_mask matches the shape of eeg_data
        min_seq_len = min(eeg_data.shape[2], past_observed_mask.shape[1])

        # Truncate both tensors to the minimum sequence length
        eeg_data = eeg_data[:, :, :min_seq_len]
        past_observed_mask = past_observed_mask[:, :min_seq_len, :]

        # Check for shape mismatches and skip if they occur
        if past_observed_mask.shape != eeg_data.shape:
            print(
                f"Skipping batch due to shape mismatch: eeg_data shape {eeg_data.shape}, "
                f"past_observed_mask shape {past_observed_mask.shape}"
            )
            continue

        # Debugging shapes
        print(f"eeg_data shape: {eeg_data.shape}")
        print(f"past_time_features shape: {past_time_features.shape}")
        print(f"past_observed_mask shape: {past_observed_mask.shape}")

        # Forward pass
        try:
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

        except Exception as e:
            print(f"Error during forward pass or optimization: {e}")
            continue

    # Print loss if it was calculated, otherwise indicate the epoch was skipped
    if loss is not None:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
    else:
        print(f"Epoch {epoch}: No valid batches processed, skipping.")

# Save the model after training
model_save_path = "/workspaces/group1-final-transformers/model_checkpoint.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")