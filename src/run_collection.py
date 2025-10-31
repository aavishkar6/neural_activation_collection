# !/usr/bin/env python3
"""
Main script for running activation collection.
"""

import argparse
import yaml
import sys
from pathlib import Path


from collectors.activation_collector import ActivationCollector
from data.dataset_loader import DatasetLoader
# from ..src.utils.memory_manager import MemoryManager
# from ..src.utils.storage import ActivationStorage

def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load command line arguments.
    parser = argparse.ArgumentParser(description='Collect neural activations from language models')

    # load config file.
    parser.add_argument(
        '--config',
        type=str,
        default='./config/base_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration.
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)


    # load strategy configurations.
    strategy_config_path = Path(args.config).parent / 'collection_strategies.yaml'
    if strategy_config_path.exists():
        with open(strategy_config_path, 'r') as f:
            strategy_configs = yaml.safe_load(f)
            config['collection']['strategies_config'] = strategy_configs.get('strategies', {})
    
    # load data
    print("Loading datasets...")
    data_loader = DatasetLoader(config)
    harmful_data, harmless_data = data_loader.load_all()

    # # Load data.
    # data_loader = DatasetLoader(config)
    # harmful_data, harmless_data = data_loader.load_all()

    print("\n" + "="*80)
    print("ACTIVATION COLLECTION")
    print("="*80)
    print(f"Run name: {config['output']['run_name']}")
    print("="*40)
    print(f"Output directory: {config['output']['base_dir']}")
    print("="*40)
    print(f"Models: {[m['name'] for m in config['models']]}")
    print("="*40)
    print(f"Categories: {config['data']['categories']}")
    print("="*40)
    print(f"Strategies: {config['collection']['strategies']}")
    print("="*40)

    # # Initialize collector
    collector = ActivationCollector(config)

    # # Collect activation values
    try:
        pass
        collector.collect_all(
            models_config = config['models'],
            data = harmful_data,
            strategies = config['collection']['strategies'],
            harmless_data = harmless_data
        )


    except Exception as e:
        print(f"Collection failed with error : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)










if __name__ == '__main__':
    main()