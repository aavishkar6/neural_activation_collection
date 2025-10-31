from .base_strategy import BaseCollectionStrategy
from .token_position_strategy import (
    TokenPositionStrategy,
    FirstTokenStrategy,
    LastTokenStrategy,
    MeanLast5Strategy,
    FirstFiveStrategy
)
# from .template_strategy import ConceptPositionStrategy

STRATEGY_REGISTRY = {
    'first_token': FirstTokenStrategy,
    'last_token': LastTokenStrategy,
    'mean_last_5': MeanLast5Strategy,
    'first_5_tokens': FirstFiveStrategy
}

def get_strategy(strategy_name: str, config: dict = None):
    """Factory function to get strategy instance"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(strategy_name, config or {})