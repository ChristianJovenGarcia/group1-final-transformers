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
    input_size=128,  # feature size
    feature_size=128,  # must match input last dim
    window_size=128,  # match context_length
    hidden_size=128,  # d_model
    d_model=128,
    num_hidden_layers=4,
    num_attention_heads=8,
    scaling="mean",
    context_length=128,  # match training
    prediction_length=32  # match training
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
dataset = TUARDataset(data_dir=dataset_path, fixed_length=128, max_files=10)
data_loader = DataLoader(dataset, batch_size=1)

# Create output directory
output_dir = "/workspaces/group1-final-transformers/denoised_outputs"
os.makedirs(output_dir, exist_ok=True)

# Create log file
log_file = "/workspaces/group1-final-transformers/inference_log.txt"

with open(log_file, "w") as log:
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            eeg_data, time_feat, past_observed_mask = batch
            if eeg_data is None or time_feat is None or past_observed_mask is None:
                print(f"Skipping batch {i} due to invalid data.")
                log.write(f"Skipping batch {i} due to invalid data.\n")
                continue

            eeg_data = eeg_data.float()
            time_feat = time_feat.float()
            past_observed_mask = past_observed_mask.float()

            # Ensure input shape: [batch, seq_len, 128]
            # Use context_length from config (128)
            seq_len = config.context_length
            # If eeg_data shape is [1, 64, 57], pad or truncate to [1, 64, 128]
            eeg_data = F.pad(eeg_data, (0, max(0, seq_len - eeg_data.shape[2])))
            eeg_data = eeg_data[:, :, :seq_len]  # [1, 64, 128]
            # Transpose to [1, 128, 64]
            eeg_data = eeg_data.permute(0, 2, 1)

            # If time_feat shape is [1, 57, N], pad/truncate to [1, 128, N]
            time_feat = F.pad(time_feat, (0, 0, 0, max(0, seq_len - time_feat.shape[1])))
            time_feat = time_feat[:, :seq_len, :]

            # Concatenate EEG and time features along last dim
            # If time_feat has no features, just use eeg_data
            if time_feat.shape[-1] > 0:
                transformer_inputs = torch.cat([eeg_data, time_feat], dim=-1)
            else:
                transformer_inputs = eeg_data

            # Project to 128 if needed
            if transformer_inputs.shape[-1] != 128:
                projection_layer = nn.Linear(transformer_inputs.shape[-1], 128).to(transformer_inputs.device)
                transformer_inputs = projection_layer(transformer_inputs)

            # Ensure input is [batch, seq_len, 128]
            # Remove any further transposes or shape changes
            print(f"transformer_inputs shape before model: {transformer_inputs.shape}")
            try:
                denoised = model(transformer_inputs)
                print(f"denoised shape after model: {denoised.shape}")
                print(f"Batch {i}: transformer_inputs shape {transformer_inputs.shape}")
                log.write(f"Batch {i}: transformer_inputs shape {transformer_inputs.shape}\n")
                log.write(f"Processed batch {i}: denoised output shape {denoised.shape}\n")
                if denoised is not None:
                    output_path = os.path.join(output_dir, f"denoised_batch_{i}.npy")
                    np.save(output_path, denoised.cpu().numpy())
                    print(f"Denoised output saved to {output_path}")
                    # denoised: [batch, seq_len, num_channels], eeg_data: [batch, seq_len, channels]
                    plt.figure(figsize=(12, 6))
                    plt.plot(eeg_data[0, :, 0].cpu().numpy(), label="Noisy EEG", alpha=0.7)
                    plt.plot(denoised[0, :, 0].cpu().numpy(), label="Denoised EEG", alpha=0.7)
                    plt.title(f"Batch {i} - Noisy vs Denoised EEG (Channel 1)")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f"denoised_batch_{i}.png"))
                    plt.close()
                else:
                    print(f"No denoised output for batch {i}.")
            except Exception as e:
                log.write(f"Error processing batch {i}: {e}\n")
                print(f"Error processing batch {i}: {e}")

print(f"Inference log saved to {log_file}")