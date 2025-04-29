import os
import mne
import numpy as np
import torch

class TUARDataset:
    def __init__(self, data_dir=None, fixed_length=57, max_files=50):  # Set fixed_length to 57
        self.data_dir = data_dir
        self.fixed_length = fixed_length
        self.max_files = max_files
        self.files = self._load_files()

    def _load_files(self):
        # Load all .edf files from the data directory
        edf_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".edf"):
                    edf_files.append(os.path.join(root, file))
        return edf_files[:self.max_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            # Load the .edf file
            raw = mne.io.read_raw_edf(file_path, preload=True)
            eeg_data, _ = raw[:, :]  # Extract data and ignore times

            # Truncate or pad data to fixed_length
            if eeg_data.shape[1] > self.fixed_length:
                eeg_data = eeg_data[:, :self.fixed_length]
            elif eeg_data.shape[1] < self.fixed_length:
                padding = self.fixed_length - eeg_data.shape[1]
                eeg_data = np.pad(eeg_data, ((0, 0), (0, padding)), mode="constant")

            # Generate past_observed_mask with the same sequence length as eeg_data
            past_observed_mask = np.ones((eeg_data.shape[0], eeg_data.shape[1]), dtype=np.float32)

            # Generate past_time_features (time_feat)
            time_feat = np.linspace(0, 1, eeg_data.shape[1])  # Ensure it matches the sequence length
            time_feat = np.tile(time_feat, (eeg_data.shape[0], 1))  # Repeat for each channel

            # Ensure shapes match
            assert eeg_data.shape == past_observed_mask.shape == time_feat.shape, (
                f"Shape mismatch: eeg_data shape {eeg_data.shape}, "
                f"past_observed_mask shape {past_observed_mask.shape}, "
                f"time_feat shape {time_feat.shape}"
            )

            # Return tensors
            return (
                torch.tensor(eeg_data, dtype=torch.float32),
                torch.tensor(time_feat, dtype=torch.float32),
                torch.tensor(past_observed_mask, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None, None