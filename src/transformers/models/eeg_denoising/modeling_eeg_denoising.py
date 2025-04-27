import torch
from transformers import TimeSeriesTransformerModel, PreTrainedModel
from .configuration_eeg_denoising import EEGDenoisingConfig

class EEGDenoisingModel(PreTrainedModel):
    config_class = EEGDenoisingConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TimeSeriesTransformerModel(config)
        self.decoder = torch.nn.Linear(config.d_model, config.num_channels)

    def forward(self, noisy_eeg, past_time_features=None, past_observed_mask=None):
        
        """
        Forward pass for the EEGDenoisingModel.

        Args:
            noisy_eeg (torch.Tensor): Input EEG data of shape (batch_size, num_channels, sequence_length).
            past_time_features (torch.Tensor, optional): Additional time features of shape (batch_size, sequence_length, feature_size).
            past_observed_mask (torch.Tensor, optional): Mask indicating observed values of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Denoised EEG data of shape (batch_size, num_channels, sequence_length).
        """
        # Reshape noisy_eeg to match the expected input format for the transformer
        batch_size, num_channels, sequence_length = noisy_eeg.shape
        transformer_input = noisy_eeg.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, num_channels)

        # Ensure observed_indicator matches the shape of data
        if past_observed_mask is not None and past_observed_mask.dim() == 2:
            past_observed_mask = past_observed_mask.unsqueeze(-1)

        # Pass inputs to the transformer
        outputs = self.transformer(
            past_values=transformer_input,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
        )
        # Access loc and scale if needed
        loc = outputs.loc
        scale = outputs.scale
        # Decode the transformer outputs to reconstruct the EEG signal
        return self.decoder(outputs.last_hidden_state.permute(0, 2, 1))  # Shape: (batch_size, num_channels, sequence_length)