from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class EEGDenoisingConfig(PretrainedConfig):
    model_type = "eeg_denoising"

    def __init__(self, **kwargs):
        # Set default hidden size to 57
        self.hidden_size = kwargs.pop("hidden_size", 57)
        self.d_model = self.hidden_size  # Ensure d_model matches hidden_size
        self.encoder_ffn_dim = kwargs.pop("encoder_ffn_dim", self.hidden_size * 4)  # Adjust feed-forward size
        self.decoder_ffn_dim = kwargs.pop("decoder_ffn_dim", self.hidden_size * 4)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 8)  # Ensure divisibility by hidden_size

        # Dynamically set attributes passed via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Fallback mechanism for missing attributes
        self._set_default_attributes()

        super().__init__(**kwargs)

        # Explicitly defined attributes with default values
        self.scaling_dim = kwargs.pop("scaling_dim", 1)
        self.keepdim = kwargs.pop("keepdim", True)
        self.minimum_scale = kwargs.pop("minimum_scale", 1e-10)
        self.default_scale = kwargs.pop("default_scale", None)
        self.num_channels = kwargs.pop("num_channels", 64)
        self.window_size = kwargs.pop("window_size", 256)
        self.use_spectral_attention = kwargs.pop("use_spectral_attention", False)  # Add this line
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 4)
        self.encoder_attention_heads = kwargs.pop("encoder_attention_heads", 8)
        self.decoder_attention_heads = kwargs.pop("decoder_attention_heads", 8)
        self.decoder_layers = kwargs.pop("decoder_layers", 4)
        self.scaling = kwargs.pop("scaling", "mean")
        self.num_static_categorical_features = kwargs.pop("num_static_categorical_features", 0)
        self.dropout = kwargs.pop("dropout", 0.1)
        self.encoder_layerdrop = kwargs.pop("encoder_layerdrop", 0.0)
        self.decoder_layerdrop = kwargs.pop("decoder_layerdrop", 0.0)
        self.activation_function = kwargs.pop("activation_function", "gelu")
        self.attention_dropout = kwargs.pop("attention_dropout", 0.1)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.1)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-5)
        self.use_cache = kwargs.pop("use_cache", True)
        self.prediction_length = kwargs.pop("prediction_length", 256)
        self.context_length = kwargs.pop("context_length", 256)
        self.input_size = kwargs.pop("input_size", 1)
        self.output_size = kwargs.pop("output_size", 1)
        self.num_decoder_layers = kwargs.pop("num_decoder_layers", 4)
        self.feature_size = kwargs.pop("feature_size", 64)
        self.encoder_layers = kwargs.pop("encoder_layers", 4)
        self.vocab_size = kwargs.pop("vocab_size", None)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.return_dict = kwargs.pop("return_dict", True)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)
        self.problem_type = kwargs.pop("problem_type", None)
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", {})
        self.label2id = kwargs.pop("label2id", {})
        self.num_labels = kwargs.pop("num_labels", 2)
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.torchscript = kwargs.pop("torchscript", False)
        self.torch_dtype = kwargs.pop("torch_dtype", None)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.task_specific_params = kwargs.pop("task_specific_params", {})
        self.gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
        self.lags_sequence = kwargs.pop("lags_sequence", [1, 2, 3, 4, 5, 6, 7])  # Default lags
        self.distribution_output = kwargs.pop("distribution_output", "normal")  # Default distribution
        self.loss = kwargs.pop("loss", "nll")  # Default loss
        self.num_parallel_samples = kwargs.pop("num_parallel_samples", 100)  # Default for sampling

        # Dynamically set any additional attributes passed via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Fallback mechanism for missing attributes
        self._set_default_attributes()

        super().__init__(**kwargs)

    def __getattr__(self, name):
        """
        Fallback for undefined attributes. Returns `None` for missing attributes.
        """
        # Log a warning for missing attributes (optional)
        if logger.isEnabledFor(logging.DEBUG):  # Log only in debug mode
            logger.warning(f"Attribute '{name}' is not explicitly defined. Returning None.")
        return None

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.update(self.__dict__)  # Include all dynamically added attributes
        return config_dict

    def _set_default_attributes(self):
        """
        Set default values for attributes that might be required but are not explicitly defined or passed.
        """
        default_attributes = {
            "init_std": 0.02,  # Default value for init_std
            "decoder_layers": 4,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 512,
            # Add other default attributes here as needed
        }
        for key, value in default_attributes.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return config