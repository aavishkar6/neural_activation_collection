from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np

class BaseCollectionStrategy(ABC):
    """
    Base class for activation collection strategies.
    """
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
    
    @abstractmethod
    def collect(
        self,
        model_wrapper,
        prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Collect activations using this strategy.
        
        Args:
            model_wrapper: Model wrapper instance
            prompts: List of prompts to process
            layers: List of layer indices (None = all layers)
        
        Returns:
            Tensor of shape [n_layers, n_prompts, hidden_dim]
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this strategy"""
        pass