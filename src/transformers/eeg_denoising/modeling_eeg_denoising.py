import torch
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerPreTrainedModel

class EEGDenoisingModel(TimeSeriesTransformerPreTrainedModel):
    config_class = EEGDenoisingConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TimeSeriesTransformerModel(config)
        self.decoder = torch.nn.Linear(config.hidden_size, config.num_channels)

    def forward(self, noisy_eeg):
        outputs = self.transformer(noisy_eeg)
        denoised = self.decoder(outputs.last_hidden_state)
        return denoised