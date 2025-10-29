from .base_model import BaseModelWrapper


class LlamaModelWrapper(BaseModelWrapper):
    """Wrapper for Llama family models"""
    
    def get_layer_module(self, layer_idx: int):
        """Get layer module for Llama"""
        return self.nnsight_model.model.layers[layer_idx]
    
    def get_num_layers(self) -> int:
        """Get number of layers in Llama"""
        return self.nnsight_model.model.config.num_hidden_layers
    
    def get_hidden_size(self) -> int:
        """Get hidden size for Llama"""
        return self.nnsight_model.model.config.hidden_size