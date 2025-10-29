from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel

class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    Provides a consistent interface for different model architectures.
    """

    def __init__(self, model_name, device_map, dtype ):
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype

        self.tokenizer = None
        self.model = None
        self.nnsight_model = None

        self._load_model()

    def _load_model(self):
        """
        Load model and tokenizer.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set padding token if None.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load with nnsight wrapper.
        self.nnsight_model = LanguageModel(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )
        
        # Also load regular model for generation/evaluation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )
        
        print(f"Loaded {self.model_name}")

    @abstractmethod
    def get_layer_module(self, layer_idx):
        """
        Get the specific module for the given layer index.
        """
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """
        Returns the number of layers in the given model.
        """
        pass

    @abstractmethod
    def get_hidden_size(self) -> int:
        """
        Returns the hidden dimension size of the model.
        """
        pass

    def tokenize(self, texts: List[str], **kwargs) -> Dict:
        """Tokenize texts"""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **kwargs
        )
    
    def cleanup(self):
        """Clean up model and free memory"""
        del self.model
        del self.nnsight_model
        torch.cuda.empty_cache()