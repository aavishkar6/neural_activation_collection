import torch
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import traceback

from models import get_model_wrapper
from strategies import get_strategy
# from utils.memory_manager import MemoryManager
# from utils.storage import ActivationStorage

class ActivationCollector:
    def __init__(self, config):
        
        self.config = config


    def collect_all(self, model_config, data, strategies, harmless_data ):
        """
        Collect activations across models and categories.
        """

        # Iterate over models.
        for model_idx, model_config in enumerate(models_config):
            model_name = model_config['name']
            model_family = model_config['family']

            print(f"Processing Model : {model_name}")

            try:
                # load model
                self.current_model = get_model_wrapper(
                    model_name = model_name,
                    family = model_family,
                    device_map = 'auto',
                    dtype = 'float16'
                )

                # Iterate over different strategies.
                for strategy_idx, strategy_name in enumerate(strategies):

                    print(f"Processing categories using strategy : {strategy_name}")

                    # Iterate over data categories.
                    for category_idx, (category, prompts) in enumerate(data.items()):
                        print(f"Iterating over category : {category}")


                        try:
                            # Collect activation for one model-strategy-category combination.
                            self._collect_single(
                                model_name = model_name,
                                category = category,
                                prompts = prompts,
                                strategy_name = strategy_name,
                                harmless_prompts = harmless_data
                            )

                        except Exception as e:
                            print(f"Error : {e}")
            except Exception as e:
                print(f"Error: {e}")
                
            self.current_model.cleanup()
            self.current_model = None

    def _collect_single(self, model_name, category, prompts, strategy_name):
        """
        Collect activations for a single model-category-strategy combination.
        """

        strategy = get_strategy(strategy_name, strategy_config)

        # Get layers to collect
        layers_config = self.config.get('collection', {}).get('layers', 'all')

        # Get batch size
        batch_size = self.config.get('collection', {}).get('batch_size', 4)

        # Optimize batch size based on memory
        activations_harmful = self._collect_batched(
            prompts = prompts,
            strategy = strategy,
            layers = layers,
            batch_size = batch_size,
            concept = category
        )

        # activations_harmless = self._collect_batched(
        #     prompts = prompts,
        #     strategy = strategy,
        #     layers = layers,
        #     batch_size = batch_size,
        #     concept = category
        # )


    def _collect_batch(self, prompts, strategy, layers, batch_size, concept):
        """
        Collect activations in batches to mamnge memory.
        """

        pass



