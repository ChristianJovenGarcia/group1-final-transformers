import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

class EEGDenoisingProcessor:
    """EEG processing pipeline for denoising transformer models"""
    
    def __init__(self, 
                 sfreq: int = 256,
                 notch_freq: Optional[float] = 60.0,
                 bandpass: Tuple[float, float] = (0.5, 50.0),
                 bad_channels: Optional[list] = None,
                 ica_components: Optional[int] = None):
        """
        Args:
            sfreq: Sampling frequency
            notch_freq: Notch filter frequency (Hz), None to disable
            bandpass: Bandpass filter range (low, high) in Hz
            bad_channels: List of bad channel names to interpolate
            ica_components: Number of ICA components for artifact removal
        """
        self.sfreq = sfreq
        self.notch_freq = notch_freq
        self.bandpass = bandpass
        self.bad_channels = bad_channels or []
        self.ica_components = ica_components

    def create_mne_raw(self, eeg_data: np.ndarray, ch_names: Optional[list] = None) -> mne.io.Raw:
        """Convert numpy array to MNE Raw object"""
        n_channels = eeg_data.shape[0]
        ch_names = ch_names or [f'EEG{i+1}' for i in range(n_channels)]
        info = mne.create_info(ch_names, self.sfreq, 'eeg')
        return mne.io.RawArray(eeg_data, info)

    def apply_standard_preprocessing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply standard preprocessing pipeline"""
        # Handle bad channels
        if self.bad_channels:
            raw.info['bads'] = [ch for ch in self.bad_channels if ch in raw.ch_names]
            if raw.info['bads']:
                raw.interpolate_bads()

        # Apply notch filter if specified
        if self.notch_freq:
            raw.notch_filter(self.notch_freq, fir_design='firwin')

        # Bandpass filtering
        raw.filter(*self.bandpass, fir_design='firwin')

        # Resample if needed
        if raw.info['sfreq'] > self.sfreq:
            raw.resample(self.sfreq, npad='auto')

        return raw

    def remove_artifacts_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply ICA for artifact removal"""
        if not self.ica_components:
            return raw

        ica = mne.preprocessing.ICA(
            n_components=self.ica_components,
            method='infomax',
            random_state=42
        )
        ica.fit(raw)
        
        # Automatically detect and remove EOG/ECG artifacts
        eog_indices, _ = ica.find_bads_eog(raw)
        ecg_indices, _ = ica.find_bads_ecg(raw)
        ica.exclude = eog_indices + ecg_indices
        
        return ica.apply(raw)

    def process(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Full processing pipeline:
        1. Convert to MNE Raw
        2. Apply standard preprocessing
        3. Artifact removal (ICA)
        4. Return cleaned data
        """
        raw = self.create_mne_raw(eeg_data)
        raw = self.apply_standard_preprocessing(raw)
        
        if self.ica_components:
            raw = self.remove_artifacts_ica(raw)
        
        return raw.get_data()

    def create_sliding_windows(self, 
                             eeg_data: np.ndarray,
                             window_size: int = 256,
                             stride: int = 128) -> torch.Tensor:
        """
        Create sliding windows for transformer input
        Args:
            eeg_data: Shape (channels, time)
            window_size: Samples per window
            stride: Stride between windows
        Returns:
            windows: Shape (n_windows, channels, window_size)
        """
        n_channels, n_times = eeg_data.shape
        windows = []
        
        for start in range(0, n_times - window_size + 1, stride):
            window = eeg_data[:, start:start + window_size]
            windows.append(window)
        
        return torch.tensor(np.stack(windows), dtype=torch.float32)

class EEGDenoisingDataset(Dataset):
    """PyTorch Dataset for EEG denoising"""
    
    def __init__(self, 
                 clean_eeg: np.ndarray,
                 processor: EEGDenoisingProcessor,
                 noise_factor: float = 0.2,
                 window_size: int = 256,
                 stride: int = 128):
        """
        Args:
            clean_eeg: Clean EEG data (channels, time)
            processor: Configured EEGDenoisingProcessor
            noise_factor: Amount of Gaussian noise to add
            window_size: Window size in samples
            stride: Stride between windows
        """
        self.clean_eeg = processor.process(clean_eeg)
        self.processor = processor
        self.noise_factor = noise_factor
        self.window_size = window_size
        self.stride = stride
        
        # Create windows
        self.clean_windows = processor.create_sliding_windows(
            self.clean_eeg, window_size, stride)
        
    def __len__(self) -> int:
        return len(self.clean_windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_windows[idx]
        noisy = clean + self.noise_factor * torch.randn_like(clean)
        return noisy, clean

def create_data_loaders(clean_eeg: np.ndarray,
                       processor: EEGDenoisingProcessor,
                       batch_size: int = 32,
                       validation_split: float = 0.2,
                       **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    Args:
        clean_eeg: Raw EEG data (channels, time)
        processor: Configured EEGDenoisingProcessor
        batch_size: Batch size
        validation_split: Fraction for validation
        **dataset_kwargs: Passed to EEGDenoisingDataset
    """
    dataset = EEGDenoisingDataset(clean_eeg, processor, **dataset_kwargs)
    
    # Split dataset
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Sample data: 19 channels, 10 seconds @ 256Hz
    sample_eeg = np.random.randn(19, 2560) * 1e-6  # Simulated EEG in volts
    
    # Initialize processor
    processor = EEGDenoisingProcessor(
        sfreq=256,
        notch_freq=60.0,
        bandpass=(1.0, 40.0),
        bad_channels=['EEG5'],  # Example bad channel
        ica_components=15
    )
    
    # Create dataloaders
    train_loader, val_loader = create_data_loaders(
        clean_eeg=sample_eeg,
        processor=processor,
        batch_size=32,
        window_size=256,
        stride=128,
        noise_factor=0.25
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    for noisy, clean in train_loader:
        print(f"Batch shapes - noisy: {noisy.shape}, clean: {clean.shape}")
        break