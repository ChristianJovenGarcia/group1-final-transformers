import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerModel, PreTrainedModel
from .configuration_eeg_denoising import EEGDenoisingConfig

class EEGDenoisingModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Ensure init_std is set in the configuration
        if not hasattr(config, "init_std"):
            config.init_std = 0.02  # Default value

        # Initialize TimeSeriesTransformer backbone
        self.transformer = TimeSeriesTransformerModel(config)
        
        # EEG-specific processing layers
        self.input_projection = nn.Linear(config.num_channels, config.d_model)
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.num_channels)
        )
        
        # Optional spectral attention layer
        if config.use_spectral_attention:
            self.spectral_attention = SpectralAttentionLayer(
                num_channels=config.num_channels,
                seq_len=config.context_length
            )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for linear and layer norm layers"""
        if isinstance(module, nn.Linear):
            # Use init_std from the configuration
            std = getattr(self.config, "init_std", 0.02)  # Default to 0.02 if init_std is missing
            if std is None:
                std = 0.02
            print(f"Initializing weights with std: {std}")  # Debugging
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, noisy_eeg, past_time_features=None, past_observed_mask=None):
        """
        Args:
            noisy_eeg: (batch_size, num_channels, sequence_length)
            past_time_features: Optional time features
            past_observed_mask: Optional mask for observed values

        Returns:
            denoised_eeg: (batch_size, num_channels, sequence_length)
        """
        # Validate input shapes
        assert noisy_eeg.ndim == 3, f"Expected noisy_eeg to have 3 dimensions, got {noisy_eeg.ndim}"
        if past_time_features is not None:
            assert past_time_features.ndim == 3, f"Expected past_time_features to have 3 dimensions, got {past_time_features.ndim}"
        if past_observed_mask is not None:
            assert past_observed_mask.ndim in [2, 3], f"Expected past_observed_mask to have 2 or 3 dimensions, got {past_observed_mask.ndim}"

        # Align past_observed_mask shape
        if past_observed_mask is not None:
            if past_observed_mask.ndim == 2:
                past_observed_mask = past_observed_mask.unsqueeze(-1)  # Add a channel dimension
            if past_observed_mask.shape[1] != noisy_eeg.shape[2]:
                # Dynamically align sequence length
                if past_observed_mask.shape[1] < noisy_eeg.shape[2]:
                    padding = noisy_eeg.shape[2] - past_observed_mask.shape[1]
                    past_observed_mask = torch.nn.functional.pad(past_observed_mask, (0, 0, 0, padding))
                elif past_observed_mask.shape[1] > noisy_eeg.shape[2]:
                    past_observed_mask = past_observed_mask[:, :noisy_eeg.shape[2], :]
            if past_observed_mask.shape[0] != noisy_eeg.shape[0]:
                raise ValueError(
                    f"Mismatch in batch size: past_observed_mask.shape[0] ({past_observed_mask.shape[0]}) "
                    f"!= noisy_eeg.shape[0] ({noisy_eeg.shape[0]})"
                )

        # Permute noisy_eeg to match the expected input shape for input_projection
        noisy_eeg = noisy_eeg.permute(0, 2, 1)  # Shape: [batch_size, sequence_length, num_channels]

        # Project noisy_eeg to the model's input dimension
        past_values = self.input_projection(noisy_eeg)  # Shape: [batch_size, sequence_length, d_model]

        # Pass inputs to the transformer
        transformer_output = self.transformer(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask
        )

        # Decode the output
        denoised_eeg = self.decoder(transformer_output.last_hidden_state)  # Shape: [batch_size, sequence_length, num_channels]

        # Permute the output back to the original shape
        denoised_eeg = denoised_eeg.permute(0, 2, 1)  # Shape: [batch_size, num_channels, sequence_length]
        return denoised_eeg

class SpectralAttentionLayer(nn.Module):
    """
    Spectral attention layer for EEG frequency band processing
    Applies attention across frequency bands derived from FFT
    """
    def __init__(self, num_channels, seq_len):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        
        # Frequency band processing
        self.query = nn.Linear(seq_len, seq_len)
        self.key = nn.Linear(seq_len, seq_len)
        self.value = nn.Linear(seq_len, seq_len)
        
        # Output projection
        self.proj = nn.Linear(seq_len, seq_len)
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size = x.shape[0]
        
        # Compute FFT
        fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        magnitude = torch.abs(fft)
        
        # Compute attention weights
        q = self.query(magnitude)
        k = self.key(magnitude)
        v = self.value(magnitude)
        
        # Scaled dot-product attention
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.seq_len ** 0.5), dim=-1)
        weighted = torch.bmm(attn, v)
        
        # Project back
        weighted = self.proj(weighted)
        
        # Inverse FFT
        modified_fft = fft * (weighted / (magnitude + 1e-6)).unsqueeze(-1)
        output = torch.fft.irfft(modified_fft, n=self.seq_len, dim=-1, norm='ortho')
        
        return output

class EEGDenoisingLoss(nn.Module):
    """
    Custom loss function for EEG denoising
    Combines MSE with spectral coherence loss
    """
    def __init__(self, mse_weight=1.0, spectral_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.mse_loss = nn.MSELoss()
        
    def spectral_loss(self, clean, denoised):
        clean_fft = torch.fft.rfft(clean, dim=-1, norm='ortho')
        denoised_fft = torch.fft.rfft(denoised, dim=-1, norm='ortho')
        
        clean_mag = torch.abs(clean_fft)
        denoised_mag = torch.abs(denoised_fft)
        
        return torch.mean(torch.abs(clean_mag - denoised_mag))
        
    def forward(self, clean, denoised):
        mse = self.mse_loss(clean, denoised)
        spec = self.spectral_loss(clean, denoised)
        return self.mse_weight * mse + self.spectral_weight * spec

if __name__ == "__main__":
    # Example usage
    config = EEGDenoisingConfig(
        num_channels=19,
        d_model=128,
        context_length=256,
        use_spectral_attention=True
    )
    
    model = EEGDenoisingModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test forward pass
    test_input = torch.randn(2, 19, 256)  # batch, channels, seq_len
    output = model(test_input)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")