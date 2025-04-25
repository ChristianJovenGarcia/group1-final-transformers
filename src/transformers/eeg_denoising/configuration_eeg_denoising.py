from transformers import PretrainedConfig

class EEGDenoisingConfig(PretrainedConfig):
    def __init__(
        self,
        num_channels=64,
        window_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads