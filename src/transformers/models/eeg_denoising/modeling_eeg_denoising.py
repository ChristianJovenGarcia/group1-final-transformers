import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerModel, PreTrainedModel
from configuration_eeg_denoising import EEGDenoisingConfig

class EEGDenoisingModel(PreTrainedModel):
    """
    EEG Denoising Transformer Model based on TimeSeriesTransformer architecture
    Takes noisy EEG inputs and outputs clean reconstructed signals
    """
    config_class = EEGDenoisingConfig

    def __init__(self, config):
        super().__init__(config)
        
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
            nn.init.xavier_uniform_(module.weight)
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
        batch_size, num_channels, sequence_length = noisy_eeg.shape
        
        # Optional spectral attention
        if hasattr(self, 'spectral_attention'):
            noisy_eeg = self.spectral_attention(noisy_eeg)
        
        # Project input to transformer dimension
        transformer_input = noisy_eeg.permute(0, 2, 1)  # (batch, seq_len, channels)
        transformer_input = self.input_projection(transformer_input)
        
        # Prepare masks if provided
        if past_observed_mask is not None:
            if past_observed_mask.ndim == 3:
                past_observed_mask = past_observed_mask.permute(0, 2, 1)
            past_observed_mask = past_observed_mask.expand_as(transformer_input)
        
        # Transformer processing
        transformer_outputs = self.transformer(
            past_values=transformer_input,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            output_hidden_states=True
        )
        
        # Decode transformer outputs
        hidden_states = transformer_outputs.last_hidden_state
        decoded = self.decoder(hidden_states)
        
        # Reshape to original EEG format
        denoised_eeg = decoded.permute(0, 2, 1)  # (batch, channels, seq_len)
        
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