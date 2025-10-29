# Implement function to get appropriate model wrapper.
from .base_model import BaseModelWrapper
from .llama_model import LlamaModelWrapper
from .gemma_model import GemmaModelWrapper
from .qwen_model import QwenModelWrapper

MODEL_REGISTRY = {
    'llama': LlamaModelWrapper,
    'gemma': GemmaModelWrapper,
    'qwen': QwenModelWrapper,
}

def get_model_wrapper(model_name: str, family: str, **kwargs) -> BaseModelWrapper:
    """
    Factory function to get appropriate model wrapper.
    
    Args:
        model_name: HuggingFace model name
        family: Model family (llama, gemma, qwen)
        **kwargs: Additional arguments for model loading
    
    Returns:
        Initialized model wrapper
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: {family}. Supported: {list(MODEL_REGISTRY.keys())}")
    
    wrapper_class = MODEL_REGISTRY[family]
    return wrapper_class(model_name, **kwargs)


__all__ = [
    'BaseModelWrapper',
    'LlamaModelWrapper',
    'GemmaModelWrapper',
    'QwenModelWrapper',
    'MODEL_REGISTRY',
    'get_model_wrapper'
]


