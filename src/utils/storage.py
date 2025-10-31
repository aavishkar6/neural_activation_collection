import os
import torch
import numpy as np
# import h5py
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

class ActivationStorage:
    """
    Manages storage of activation tensors with proper organization.
    
    Directory structure:
        base_dir/
            run_001/
                metadata.json
                model_llama-7b/
                    strategy_last_token/
                        category_Economic_harm.pt
                        category_Violence.pt
                    strategy_first_token/
                        ...
                model_gemma-2b/
                    ...
    """

    def __init__(
        self,
        base_dir: str,
        run_name: str,
        save_format: str = "pt",
        compression: bool = True
    ):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.save_format = save_format
        self.compression = compression
        
        # Create run directory
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            'run_name': run_name,
            'created_at': datetime.now().isoformat(),
            'save_format': save_format,
            'compression': compression,
            'models': {},
            'categories': {},
            'strategies': {}
        }

    def get_save_path(self, model_name, category, strategy):
        """
        Get save path for a specific combination.
        """

        # Clean names for filesystem
        model_clean = model_name.replace('/', '_').replace(' ', '_')
        category_clean = category.replace('/', '_').replace(' ', '_')
        strategy_clean = strategy.replace('/', '_').replace(' ', '_')

        # Create category
        category_dir = self.run_dir / f"model_{model_clean}" / f"strategy_{strategy_clean}"
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = f"category_{category_clean}.{self.save_format}"

        return category_dir / filename

    def save_activations(self, activations, model_name, category, strategy):

        save_path = self.get_save_path(model_name, category, strategy)

        print(f"Save path is {save_path}")

        torch.save(activations, save_path)

        print(f"✓ Saved: {save_path.relative_to(self.base_dir)}")


    