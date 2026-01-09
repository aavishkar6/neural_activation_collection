import torch
from datasets import load_dataset
from typing import Dict, List, Optional
from pathlib import Path
import json

class DatasetLoader:
    """
    Handles loading and preprocessing of datasets.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get('data', {})
    
    def load_harmful_data(self) -> Dict[str, List[str]]:
        """
        Load harmful prompts organized by category.
        
        Returns:
            Dict mapping category -> list of prompts
        """
        dataset_name = self.data_config.get('dataset_name', 'declare-lab/CategoricalHarmfulQA')
        categories = self.data_config.get('categories', [])
        num_samples = self.data_config.get('num_samples_per_category', 50)
        
        print(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Convert to pandas for easier filtering
        if 'en' in dataset:
            df = dataset['en'].to_pandas()
        elif 'train' in dataset:
            df = dataset['train'].to_pandas()
        else:
            # Use first available split
            split_name = list(dataset.keys())[0]
            df = dataset[split_name].to_pandas()
        
        # Organize by category
        data = {}
        for category in categories:
            # Filter by category
            category_df = df[df['Category'] == category]
            
            # Get prompts (adjust column name based on dataset)
            if 'Question' in category_df.columns:
                prompts = category_df['Question'].to_list()
            elif 'prompt' in category_df.columns:
                prompts = category_df['prompt'].to_list()
            elif 'text' in category_df.columns:
                prompts = category_df['text'].to_list()
            else:
                # Try to find the right column
                text_columns = [col for col in category_df.columns if 'text' in col.lower() or 'question' in col.lower() or 'prompt' in col.lower()]
                if text_columns:
                    prompts = category_df[text_columns[0]].to_list()
                else:
                    raise ValueError(f"Could not find text column in dataset for category {category}")
            
            # Limit to num_samples
            prompts = prompts[:num_samples]
            
            data[category] = prompts
            print(f"  ✓ Loaded {len(prompts)} prompts for category: {category}")
        
        return data
    
    def load_harmless_data(self) -> List[str]:
        """
        Load harmless/benign prompts.
        
        Returns:
            List of harmless prompts
        """
        harmless_path = self.data_config.get('harmless_data_path')
        num_samples = self.data_config.get('num_samples_per_category', 50)
        
        if harmless_path is None:
            print("Warning: No harmless data path provided. Using default prompts.")
            # Return some default harmless prompts
            return self._get_default_harmless_prompts(num_samples)
        
        harmless_path = Path(harmless_path)
        
        if not harmless_path.exists():
            print(f"Warning: Harmless data file not found at {harmless_path}. Using defaults.")
            return self._get_default_harmless_prompts(num_samples)
        
        # Load from file
        with open(harmless_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        # Limit to num_samples
        prompts = prompts[:num_samples]  # Get extra for flexibility
        
        print(f"  ✓ Loaded {len(prompts)} harmless prompts")
        
        return prompts

    def load_harmful_data_rewritten_prompts(self) -> Dict[str, List[str]]:
        """
        Load the rewritten data that were received by prompting mistral.
        """

        self.rewritten_prompts_data_path = self.data_config.get('rewritten_prompts_data_path', {})
        assert self.rewritten_prompts_data_path is not None, "Dataset path is empty. Please provide path to the dataset under config."

        # Load the json file.
        with open(self.rewritten_prompts_data_path, 'r') as file:
            data = json.load(file)

        harmful_data = {}

        # Loop over all the keys.
        for categories, data_points in data.items():
            row = []
            # loop over each of the harmful categories.
            for prompts in data_points:
                row.extend( [prompts['original_prompt']] + prompts['rewritten_prompts'])
            # Append to a dictionary.
            harmful_data[categories] = row

        # Return the data as a dict.
        return harmful_data

    def load_all(self) -> tuple:
        """
        Load all data.
        
        Returns:
            Tuple of (harmful_data, harmless_data)
        """
        harmful_data = self.load_harmful_data_rewritten_prompts()
        harmless_data = self.load_harmless_data()
        
        return harmful_data, harmless_data