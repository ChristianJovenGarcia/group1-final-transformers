import os
import mne
import numpy as np
import torch

class TUARDataset:
    def __init__(self, data_dir=None, fixed_length=20000, max_files=50):
        self.data_dir = data_dir or "/workspaces/group1-final-transformers/examples/research_projects/eeg_denoising/data"  # Default path
        self.fixed_length = fixed_length
        self.max_files = max_files  # Maximum number of files to process

        if not os.path.exists(self.data_dir):
            raise ValueError(f"Dataset directory does not exist: {self.data_dir}")

        # Recursively find all .edf files in the directory and subdirectories
        self.files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".edf"):
                    self.files.append(os.path.join(root, file))

        if len(self.files) == 0:
            raise ValueError(f"No .edf files found in dataset directory: {self.data_dir}")

        # Limit the number of files to process
        self.files = self.files[:self.max_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            # Load the .edf file corresponding to the given index
            raw = mne.io.read_raw_edf(file_path, preload=True)
            data, _ = raw[:, :]  # Extract data and ignore times

            # Pad or truncate the data to the fixed length
            if data.shape[1] < self.fixed_length:
                padded_data = np.zeros((data.shape[0], self.fixed_length))
                padded_data[:, :data.shape[1]] = data
                data = padded_data
            elif data.shape[1] > self.fixed_length:
                data = data[:, :self.fixed_length]

            # Generate dummy past_time_features and past_observed_mask
            past_time_features = np.zeros((self.fixed_length, 1))  # Example: all zeros
            past_observed_mask = np.ones((data.shape[0], self.fixed_length))  # Example: all ones

            # Ensure past_observed_mask matches the shape of data
            past_observed_mask = past_observed_mask[:data.shape[0], :data.shape[1]]

            # Convert to PyTorch tensors
            data = torch.tensor(data, dtype=torch.float32)
            past_time_features = torch.tensor(past_time_features, dtype=torch.float32)
            past_observed_mask = torch.tensor(past_observed_mask, dtype=torch.float32)

            return data, past_time_features, past_observed_mask

        except Exception as e:
            # Log the error and skip the file
            print(f"Error processing file {file_path}: {e}")
            return None, None, None