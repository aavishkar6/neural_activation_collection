from .base_model import BaseModelWrapper

class QwenModelWrapper(BaseModelWrapper):
    """Wrapper for Qwen family models"""
    
    def get_layer_module(self, layer_idx: int):
        """Get layer module for Qwen"""
        # Qwen uses 'transformer.h' instead of 'model.layers'
        return self.nnsight_model.transformer.h[layer_idx]
    
    def get_num_layers(self) -> int:
        """Get number of layers in Qwen"""
        return self.nnsight_model.config.num_hidden_layers
    
    def get_hidden_size(self) -> int:
        """Get hidden size for Qwen"""
        return self.nnsight_model.config.hidden_size