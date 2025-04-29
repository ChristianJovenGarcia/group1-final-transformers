import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers.models.eeg_denoising.modeling_eeg_denoising import EEGDenoisingModel
from transformers.models.eeg_denoising.configuration_eeg_denoising import EEGDenoisingConfig
from utils import TUARDataset
from torch.utils.data import DataLoader
import psutil
import torch.nn.functional as F
import torch.nn as nn

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

# Load the model configuration
config = EEGDenoisingConfig(
    num_channels=64,
    input_size=128,      # <--- CHANGE THIS TO 128
    feature_size=128,    # <--- CHANGE THIS TO 128
    window_size=57,
    hidden_size=128,    # <--- CHANGE THIS TO 128
    num_hidden_layers=4,
    num_attention_heads=8,
    scaling="mean",
    d_model=128         # <--- CHANGE THIS TO 128
)

# Initialize the model
model = EEGDenoisingModel(config)

# Load the saved model weights
model_load_path = "/workspaces/group1-final-transformers/model_checkpoint.pth"
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode
print("Model loaded and ready for inference.")

# Initialize the dataset and data loader
dataset_path = "/workspaces/group1-final-transformers/examples/research_projects/eeg_denoising/data/physionet.org/files/eegmmidb/1.0.0/"
dataset = TUARDataset(data_dir=dataset_path, fixed_length=57, max_files=10)
data_loader = DataLoader(dataset, batch_size=1)

# Create output directory
output_dir = "/workspaces/group1-final-transformers/denoised_outputs"
os.makedirs(output_dir, exist_ok=True)

# Create log file
log_file = "/workspaces/group1-final-transformers/inference_log.txt"

with open(log_file, "w") as log:
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            eeg_data, time_feat, past_observed_mask = batch  # Ensure time_feat is unpacked
            if eeg_data is None or time_feat is None or past_observed_mask is None:
                print(f"Skipping batch {i} due to invalid data.")
                log.write(f"Skipping batch {i} due to invalid data.\n")
                continue

            eeg_data = eeg_data.float()
            time_feat = time_feat.float()
            past_observed_mask = past_observed_mask.float()

            # Debugging shapes
            print(f"Original past_observed_mask shape: {past_observed_mask.shape}")

            # Ensure past_observed_mask has exactly 3 dimensions
            if past_observed_mask.ndim == 2:  # Add a new dimension only if it's 2D
                past_observed_mask = past_observed_mask.unsqueeze(-1)

            # Ensure past_observed_mask matches the shape of context
            past_observed_mask = past_observed_mask[:, :57, :57]  # Truncate to match the sequence length
            padding_size = 128 - past_observed_mask.shape[-1]  # Calculate padding size
            past_observed_mask = F.pad(past_observed_mask, (0, padding_size))  # Pad to match hidden size
            print(f"After padding: {past_observed_mask.shape}")

            # Debugging shapes
            print(f"observed_context shape: {past_observed_mask.shape}")  # Replace 'context' with 'past_observed_mask'

            # Define aligned_time_feat
            aligned_time_feat = time_feat[:, :57, :]  # Truncate or process time_feat to match the expected shape
            print(f"aligned_time_feat shape: {aligned_time_feat.shape}")

            # Define aligned_expanded_static_feat
            aligned_expanded_static_feat = torch.zeros_like(aligned_time_feat)  # Placeholder for static features
            print(f"aligned_expanded_static_feat shape: {aligned_expanded_static_feat.shape}")

            # Define transformer_inputs by concatenating features
            transformer_inputs = torch.cat([aligned_time_feat, aligned_expanded_static_feat], dim=-1)  # Remove reshaped_lagged_sequence
            print(f"transformer_inputs shape: {transformer_inputs.shape}")

            # Project to match the checkpoint's expected input size
            expected_dim = 128  # Match the checkpoint's input size
            if transformer_inputs.shape[-1] != expected_dim:
                print(f"Projecting transformer_inputs from {transformer_inputs.shape[-1]} to {expected_dim}")
                projection_layer = nn.Linear(transformer_inputs.shape[-1], expected_dim).to(transformer_inputs.device)
                transformer_inputs = projection_layer(transformer_inputs)
                print(f"transformer_inputs shape after projection: {transformer_inputs.shape}")

            try:
                # Perform inference
                denoised = model(transformer_inputs)

                # Debugging shapes
                print(f"Batch {i}: transformer_inputs shape {transformer_inputs.shape}")
                log.write(f"Batch {i}: transformer_inputs shape {transformer_inputs.shape}\n")

                log.write(f"Processed batch {i}: denoised output shape {denoised.shape}\n")

                # Save the denoised output
                if denoised is not None:
                    output_path = os.path.join(output_dir, f"denoised_batch_{i}.npy")
                    np.save(output_path, denoised.cpu().numpy())  # Save as .npy file
                    print(f"Denoised output saved to {output_path}")

                    # Permute for plotting: [batch, channels, seq_len]
                    denoised_for_plot = denoised.permute(0, 2, 1)  # [batch, channels, seq_len]
                    # Plot the first channel of the first sample
                    plt.figure(figsize=(12, 6))
                    plt.plot(eeg_data[0, 0, :].cpu().numpy(), label="Noisy EEG", alpha=0.7)
                    plt.plot(denoised_for_plot[0, 0, :].cpu().numpy(), label="Denoised EEG", alpha=0.7)
                    plt.title(f"Batch {i} - Noisy vs Denoised EEG (Channel 1)")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f"denoised_batch_{i}.png"))  # Save the plot
                    plt.close()
                else:
                    print(f"No denoised output for batch {i}.")
            except Exception as e:
                log.write(f"Error processing batch {i}: {e}\n")
                print(f"Error processing batch {i}: {e}")

print(f"Inference log saved to {log_file}")