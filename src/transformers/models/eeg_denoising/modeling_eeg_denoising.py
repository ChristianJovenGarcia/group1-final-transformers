import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerModel, PreTrainedModel
from .configuration_eeg_denoising import EEGDenoisingConfig

class EEGDenoisingModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize the transformer
        self.transformer = TimeSeriesTransformerModel(config)
        
        # Input projection
        self.input_projection = nn.Linear(128, 128)  # 128 -> 128
        
        # Decoder for output
        # Fix: decoder input dim should match transformer output last dim (128), output dim = num_channels (64)
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),  # 128 -> 128
            nn.GELU(),
            nn.Linear(128, config.num_channels)  # 128 -> 64
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

    def forward(self, transformer_inputs):
        # transformer_inputs: [batch, seq_len, 128]
        if transformer_inputs.shape[2] != 128:
            raise ValueError(f"Expected last dim 128, got {transformer_inputs.shape}")
        projected_inputs = self.input_projection(transformer_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=projected_inputs,
            return_dict=True
        )
        sequence_output = transformer_outputs.last_hidden_state
        # Ensure always 3D: [batch, seq_len, 128]
        if sequence_output.dim() == 2:
            sequence_output = sequence_output.unsqueeze(0)
        # Decoder expects [batch*seq_len, 128]
        batch, seq_len, feat = sequence_output.shape
        x = sequence_output.contiguous().view(-1, feat)
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        denoised_eeg = x.view(batch, seq_len, self.config.num_channels)
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
        d_model=64,
        context_length=256,
        use_spectral_attention=True
    )
    
    model = EEGDenoisingModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test forward pass
    test_input = torch.randn(2, 19, 256)  # batch, channels, seq_len
    output = model(test_input)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")