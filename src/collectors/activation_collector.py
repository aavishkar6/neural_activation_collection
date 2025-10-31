import torch
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import traceback

from models import get_model_wrapper
from strategies import get_strategy
# from utils.memory_manager import MemoryManager
from utils.storage import ActivationStorage

class ActivationCollector:
    def __init__(self, config):
        
        self.config = config

        # Initialize storage
        output_config = config.get('output', {})

        self.storage = ActivationStorage(
            base_dir = output_config.get('base_dir', './activations_output'),
            run_name = output_config.get('run_name', 'run_001'),
            save_format = output_config.get('save_format', 'pt'),
            compression = output_config.get('compression', True),
        )

        self.current_model = None


    def collect_all(self, models_config, data, strategies, harmless_data ):
        """
        Collect activations across models and categories.
        """

        # Iterate over models.
        for model_idx, model_config in enumerate(models_config):
            model_name = model_config['name']
            model_family = model_config['family']

            print("="*50)
            print(f"Processing Model : {model_name}")
            print("="*50)

            try:
                # load model
                self.current_model = get_model_wrapper(
                    model_name = model_name,
                    family = model_family,
                    device_map = 'auto', # Take the value from model configs later update. 
                    dtype = 'float16'   # same as above comment.
                )

                # Iterate over different strategies.
                for strategy_idx, strategy_name in enumerate(strategies):

                    print("="*50)
                    print(f"    Processing categories using strategy : {strategy_name}")
                    print("="*50)

                    # Iterate over data categories.
                    for category_idx, (category, prompts) in enumerate(data.items()):
                        print(f"    Iterating over category : {category}")

                        try:
                            # Collect activation for one model-strategy-category combination.
                            activations = self._collect_single(
                                model_name = model_name,
                                category = category,
                                prompts = prompts,
                                strategy_name = strategy_name,
                            )
                            
                            self.storage.save_activations(activations, 
                                model_name, 
                                category,
                                strategy_name
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

        strategy_config = self.config.get('collection', {}).get('strategies_config', {}).get(strategy_name, {})
        strategy = get_strategy(strategy_name, strategy_config)

        # Get layers to collect
        layers_config = self.config.get('collection', {}).get('layers', 'all')
        if layers_config == 'all':
            layers = None
        else:
            layers = layers_config

        # Get batch size
        batch_size = self.config.get('collection', {}).get('batch_size', 4)


        print(f"        Collecting activations using {strategy_name} for layers : {layers_config} and batch size {batch_size}")
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

        return activations_harmful


    def _collect_batched(self, prompts, strategy, layers, batch_size, concept):
        """
        Collect activations in batches to mamnge memory.

        Returns:
            Tensor of shape [n_layers, n_prompts, hidden_dim]
        """

        n_prompts = len(prompts)
        n_batches = (n_prompts + batch_size - 1) // batch_size
        
        all_activations = []

        for batch_idx in tqdm(range(n_batches), desc="  Batches", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_prompts)
            batch_prompts = prompts[start_idx:end_idx]

            try:
                activations = self._collect_batch(
                    batch_prompts,
                    strategy,
                    layers
                )

                all_activations.append(activations)

            except Exception as e:
                print(f"      ✗ Error in batch {batch_idx}: {str(e)}")
                raise

        final_activations = torch.cat(all_activations, dim=1)

        print("     Shape of final activations ", final_activations.shape)
        return final_activations

    def _collect_batch(self, batch_prompts, strategy, layers):
        """
        Collect activations for a single batch.
        """

        activations = strategy.collect(
            self.current_model,
            batch_prompts,
            layers
        )

        # print("Shape of activations is : ", activations.shape)
        return activations





