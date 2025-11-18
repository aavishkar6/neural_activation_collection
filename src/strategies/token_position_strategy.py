import torch
from typing import List, Dict, Optional
from .base_strategy import BaseCollectionStrategy

class TokenPositionStrategy(BaseCollectionStrategy):
    """
    Strategy for collecting activations at specific token positions.
    """
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.position = config.get('position')
        self.aggregation = config.get('aggregation', None)
    
    def collect(
        self,
        model_wrapper,
        prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Collect activations at specified token position(s).
        """
        if layers is None:
            layers = list(range(model_wrapper.get_num_layers()))
        
        n_layers = len(layers)
        n_prompts = len(prompts)
        n_positions = len(self.position)
        hidden_size = model_wrapper.get_hidden_size()
        
        # Initialize output tensor
        activations = torch.zeros(n_layers, n_prompts, n_positions, hidden_size)
        
        # Use nnsight to collect activations
        with model_wrapper.nnsight_model.trace(prompts):
            for idx, layer_idx in enumerate(layers):
                # Get layer output
                layer_output = model_wrapper.get_layer_output(layer_idx)[0]
                
                # Extract positions
                if isinstance(self.position, int):
                    # Single position
                    position_acts = layer_output[:, self.position, :]
                
                elif isinstance(self.position, list):
                    # Multiple positions
                    position_acts = layer_output[:, self.position, :]
                    
                    # Aggregate if specified
                    if self.aggregation == 'mean':
                        position_acts = position_acts.mean(dim=1)
                    elif self.aggregation == 'max':
                        position_acts = position_acts.max(dim=1)[0]
                    elif self.aggregation == 'null':
                        pass
                    else:
                        raise ValueError(f"Unknown aggregation: {self.aggregation}")
                
                else:
                    raise ValueError(f"Invalid position type: {type(self.position)}")
                
                # Save activation
                activations[idx] = position_acts.save()
        
        return activations
    
    def get_description(self) -> str:
        pos_str = f"position {self.position}" if isinstance(self.position, int) else f"positions {self.position}"
        agg_str = f" with {self.aggregation} aggregation" if self.aggregation else ""
        return f"Collect activations at {pos_str}{agg_str}"


class LastTokenStrategy(TokenPositionStrategy):
    """Optimized strategy for last token (most common)"""
    
    def __init__(self, name: str = "last_token", config: Dict = None):
        config = config or {'position': -1, 'aggregation': None}
        super().__init__(name, config)


class FirstTokenStrategy(TokenPositionStrategy):
    """Strategy for first token"""
    
    def __init__(self, name: str = "first_token", config: Dict = None):
        config = config or {'position': 0, 'aggregation': None}
        super().__init__(name, config)


class MeanLast5Strategy(TokenPositionStrategy):
    """Strategy for mean of last 5 tokens"""
    
    def __init__(self, name: str = "mean_last_5", config: Dict = None):
        config = config or {'position': [-5, -4, -3, -2, -1], 'aggregation': 'mean'}
        super().__init__(name, config)

class FirstFiveStrategy(TokenPositionStrategy):
    """Strategy for the first 5 tokens"""

    def __init__(self, name: str="first_5_tokens", config: Dict = None):
        config = config or {'position': [0, 1, 2, 3, 4,], 'aggregation': 'mean'}
        super().__init__(name, config)

class FirstTenStrategy(TokenPositionStrategy):
    """Strategy for first 10 tokens"""

    def __init__(self, name: str="first_10_tokens", config: Dict = None):
        config = config or {'position': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'aggregation': 'mean'}
        super().__init__(name, config)