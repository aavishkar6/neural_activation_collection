# Neural Activations Toolkit

A comprehensive toolkit for collecting neural activations from large language models. It has support for multiple architectures and collection strategies.

## Features

- **Multi-Model Support**: Llama, Gemma, Qwen, and easily extensible to other architectures.
- **Multiple Collection Strategies**: First token, last token, concept position, and more.
- **HPC-Ready**: SLURM scripts and checkpointing for long-running jobs.
- **Organized Storage**: Hierarchical storage by run/model/category


## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/neural-activations-toolkit.git
cd neural-activations-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

```

## Quick Start

### 1. Configure Your Run

Edit `config/base_config.yaml`:
```yaml
output:
  run_name: "my_first_run"
  base_dir: "./activations_output"

models:
  - name: "meta-llama/Llama-2-7b-hf"
    family: "llama"

data:
  categories:
    - "Economic Harm"
    - "Violence"

collection:
  strategies:
    - "last_token"
    - "concept_position"
```

### 2. Run Locally
```bash
python scripts/run_collection.py --config config/base_config.yaml
```

### 3. Run on HPC
```bash
# Submit job
./scripts/submit_job.sh

# Check status
squeue -u $USER
```

## Collection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `first_token` | First token activation | Initial processing |
| `first_5_tokens` | Mean of first 5 tokens | Early context |
| `last_token` | Last token activation | **Most common - full context** |
| `mean_last_5` | Mean of last 5 tokens | Stable representation |
| `concept_position` | Activation at concept token | Template-based analysis |

## Output Structure
```
activations_output/
  run_001/
    metadata.json
    model_llama-7b/
        strategy_last_token.pt
            category_Economic_Harm/
            category_Violence/
        strategy_concept_position.pt
        ...
    model_gemma-2b/
      ...
```

## Adding a New Model

1. Create model wrapper in `src/models/`:
```python
from .base_model import BaseModelWrapper

class MyModelWrapper(BaseModelWrapper):
    def get_layer_module(self, layer_idx: int):
        return self.nnsight_model.model.layers[layer_idx]  # Adjust path
    
    def get_num_layers(self) -> int:
        return self.nnsight_model.model.config.num_hidden_layers
    
    def get_hidden_size(self) -> int:
        return self.nnsight_model.model.config.hidden_size
```

2. Register in `src/models/__init__.py`:
```python
MODEL_REGISTRY = {
    'llama': LlamaModelWrapper,
    'gemma': GemmaModelWrapper,
    'qwen': QwenModelWrapper,
    'mymodel': MyModelWrapper,  # Add here
}
```

3. Use in config:
```yaml
models:
  - name: "org/my-model"
    family: "mymodel"
```

## Adding a New Strategy

1. Create strategy in `src/strategies/`:
```python
from .base_strategy import BaseCollectionStrategy

class MyStrategy(BaseCollectionStrategy):
    def collect(self, model_wrapper, prompts, layers=None):
        # Your collection logic
        return activations
    
    def get_description(self):
        return "My custom strategy"
```

2. Register in `src/strategies/__init__.py`:
```python
STRATEGY_REGISTRY = {
    'my_strategy': MyStrategy,
}
```

3. Add to config:
```yaml
collection:
  strategies:
    - "my_strategy"
```

## Memory Management

To be added.

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size in config:
```yaml
   collection:
     batch_size: 2  # Reduce from 4
```

2. Process fewer samples:
```yaml
   data:
     num_samples_per_category: 20  # Reduce from 50
```

3. Use specific layers instead of all:
```yaml
   collection:
     layers: [15, 20, 25]  # Instead of "all"
```

### Model Loading Issues

Ensure you have HuggingFace token for gated models:
```bash
huggingface-cli login
```

### SLURM Job Failures

Check logs:
```bash
cat logs/activation_.err
```

Increase time limit:
```bash
#SBATCH --time=48:00:00
```