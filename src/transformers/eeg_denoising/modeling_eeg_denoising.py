import torch
from transformers import TimeSeriesTransformerModel, PreTrainedModel
from .configuration_eeg_denoising import EEGDenoisingConfig

class EEGDenoisingModel(PreTrainedModel):
    config_class = EEGDenoisingConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TimeSeriesTransformerModel(config)
        self.decoder = torch.nn.Linear(config.hidden_size, config.num_channels)

    def forward(self, noisy_eeg):
        outputs = self.transformer(noisy_eeg)
        return self.decoder(outputs.last_hidden_state)