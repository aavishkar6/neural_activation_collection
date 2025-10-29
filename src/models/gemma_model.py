from .base_model import BaseModelWrapper

class GemmaModelWrapper(BaseModelWrapper):
    """Wrapper for Gemma family models"""
    
    def get_layer_module(self, layer_idx: int):
        """Get layer module for Gemma"""
        return self.nnsight_model.model.layers[layer_idx]
    
    def get_num_layers(self) -> int:
        """Get number of layers in Gemma"""
        return self.nnsight_model.model.config.num_hidden_layers
    
    def get_hidden_size(self) -> int:
        """Get hidden size for Gemma"""
        return self.nnsight_model.model.config.hidden_size