# VLM Geometry: Visual Language Model Evaluation Framework

A comprehensive framework for evaluating Vision-Language Models (VLMs) on geometric reasoning tasks using stimuli generated from symbolic Domain-Specific Languages (DSLs). This codebase creates oddball detection tasks where VLMs must identify visual outliers in sets of algorithmically generated geometric patterns.

## What This Repository Does

This framework generates visual stimuli from symbolic programs and evaluates VLMs on their ability to identify visual differences across abstract dimensions. The core workflow:

1. **Generate Programs**: Create symbolic programs using geometric DSLs that specify abstract visual features
2. **Render Stimuli**: Convert programs to images using different rendering backends (deterministic, probabilistic, colored)
3. **Create Oddball Trials**: Generate test sets where 5 stimuli share a feature and 1 differs (the "oddball")
4. **Run VLM Inference**: Test various VLMs on oddball detection with different reasoning prompts
5. **Analyze Results**: Calculate accuracy and analyze performance across abstract dimensions

## Project Structure

```
vlm_geometry/
├── config/                    # Hydra configuration files
│   ├── model/                # Model configurations by provider
│   │   ├── anthropic/        # Claude models (Sonnet, Opus)
│   │   ├── openai/          # OpenAI models (GPT-5, O3, O4-mini)
│   │   └── google/          # Google models (Gemini Pro/Flash)
│   ├── task/                # Task configurations
│   └── paths/               # Path configurations
├── models/                  # Model implementations
├── tasks/                   # Task definitions
├── renderer/                # Rendering engine
│   ├── core.py             # Core parsing and rendering
│   └── languages/          # Different DSL renderers
├── prompts/                 # Prompt templates
├── run_task.py             # Main execution script
└── api_metadata.json       # API credentials
```

## Supported Models

The framework supports multiple VLM providers through a unified interface:

### Anthropic Models
- **Claude Sonnet**: `config/model/anthropic/sonnet.yaml`
- **Claude Opus**: `config/model/anthropic/opus.yaml`

### OpenAI Models
- **GPT-5**: `config/model/openai/gpt-5.yaml`
- **GPT-5 Mini**: `config/model/openai/gpt-5-mini.yaml`
- **GPT-5 Nano**: `config/model/openai/gpt-5-nano.yaml`
- **O3**: `config/model/openai/o3.yaml`
- **O4 Mini**: `config/model/openai/o4-mini.yaml`

### Google Models
- **Gemini Pro**: `config/model/google/gemini-pro.yaml`
- **Gemini Flash**: `config/model/google/gemini-flash.yaml`

### Adding New Models

1. **Create model configuration**: Add a new YAML file in `config/model/<provider>/`:
   ```yaml
   defaults:
     - base_vlm
   _target_: models.<provider>_model.<ProviderModel>
   model_name: your_model_name
   api_model_identifier: actual_api_model_name  # if different
   cost_per_input_token: 1.0    # per million tokens
   cost_per_output_token: 5.0   # per million tokens
   # Model-specific parameters
   reasoning_effort: low        # For OpenAI reasoning models
   n_thinking_tokens: 1024     # For Anthropic thinking tokens
   ```

2. **Add API credentials**: Update `api_metadata.json`:
   ```json
   {
     "your_model_name": {
       "api_key": "your_api_key",
       "endpoint": "https://api.provider.com/v1/endpoint"
     }
   }
   ```

3. **Implement model class** (if new provider): Create `models/your_provider_model.py`:
   ```python
   from models.base_model import APIModel

   class YourProviderModel(APIModel):
       def _prepare_header(self) -> dict:
           return {"Authorization": f"Bearer {self.api_key}"}

       def _prepare_endpoint(self, endpoint: str) -> str:
           return endpoint

       def build_vlm_payload(self, trial_metadata) -> dict:
           # Implement provider-specific payload format
           pass

       def _parse_response(self, response_json: dict):
           # Extract response text, answer, and token counts
           pass
   ```

## Task System

Tasks define how to generate visual stimuli and abstract features for oddball detection.

### Currently Implemented Tasks

1. **Spirographs** (`tasks/spirographs.py`): Generates mandala-like patterns with varying:
   - Central primitive shapes (circle, square, triangle, pentagon, hexagon)
   - Radial primitive shapes and arrangements
   - Scales and radii

2. **Totems** (`tasks/totems.py`): Creates totem pole-like vertical arrangements

3. **Glyphs** (`tasks/glyphs.py`): Generates abstract symbolic patterns

4. **Nuts and Bolts** (`tasks/nuts_and_bolts.py`): Creates mechanical-style geometric forms

### Task Implementation

All tasks inherit from `tasks.base_task.Task` and must implement:

```python
class MyTask(Task):
    renderer = SomeRenderer()  # Choose renderer type

    def generate_programs(self) -> pd.DataFrame:
        """Generate programs with their abstract features."""
        return pd.DataFrame({
            'program_string': ['(C l c)', '(T l s)', ...],
            'feature1': [val1, val2, ...],
            'feature2': [val1, val2, ...],
            # ... other abstract dimensions
        })
```

The base class automatically handles:
- Rendering programs to images using `renderer/core.py`
- Creating oddball trials for each abstract dimension
- Generating labeled trial images and visualizations
- Saving metadata and trial information

### Adding New Tasks

1. **Create task class**: Implement in `tasks/your_task.py`:
   ```python
   from tasks.base_task import Task
   from renderer.languages.deterministic import DeterministicRenderer

   class YourTask(Task):
       renderer = DeterministicRenderer()

       def generate_programs(self):
           # Generate programs with abstract feature columns
           return pd.DataFrame(records)
   ```

2. **Create task configuration**: Add `config/task/your_task.yaml`:
   ```yaml
   defaults:
     - base_task
   _target_: tasks.your_task.YourTask
   task_name: your_task
   ```

## Rendering System

The framework uses a modular rendering system with different backends:

- **DeterministicRenderer** (`renderer/languages/deterministic.py`): Clean geometric shapes
- **ProbabilisticRenderer** (`renderer/languages/probabilistic.py`): Adds controlled noise/variation
- **ColoredRenderer** (`renderer/languages/colored.py`): Supports color variations

All renderers share a common DSL syntax defined in `renderer/core.py`.

## Running Inference

### Basic Usage

Run a specific model and task combination:

```bash
python run_task.py model=anthropic/sonnet task=spirographs
```

### Configuration Options

The framework uses Hydra for configuration management. Key parameters:

- **Model Selection**: `model=provider/model_name`
- **Task Selection**: `task=task_name`
- **Prompt Conditions**: `model.prompt_condition=CoT` or `model.prompt_condition=no_CoT`
- **Token Budgets**: `model.max_tokens=8192`
- **Reasoning Tokens**: `model.n_thinking_tokens=2048` (Anthropic)
- **Reasoning Effort**: `model.reasoning_effort=high` (OpenAI)

### Prompt Conditions

Two main prompting strategies are supported:

1. **Chain-of-Thought (CoT)**: Asks for reasoning and justification
   ```
   Identify the outlier by responding with its number in square brackets (e.g., [3])
   and a brief justification of your choice enclosed in curly brackets.
   ```

2. **No Chain-of-Thought (no_CoT)**: Direct answer only
   ```
   Respond immediately with the number of odd-one-out in square brackets (e.g., [3])
   with no additional reasoning or justification.
   ```

### Advanced Usage Examples

Run with different reasoning token budgets:
```bash
# Low reasoning budget (Anthropic)
python run_task.py model=anthropic/sonnet model.n_thinking_tokens=512

# High reasoning budget (Anthropic)
python run_task.py model=anthropic/sonnet model.n_thinking_tokens=4096

# Different reasoning efforts (OpenAI)
python run_task.py model=openai/o3 model.reasoning_effort=low
python run_task.py model=openai/o3 model.reasoning_effort=high
```

Run with custom prompts:
```bash
python run_task.py model=anthropic/sonnet model.prompt_condition=no_CoT
```

Generate more trials:
```bash
python run_task.py task=spirographs task.n_trials=500
```

### Output Structure

Results are saved to `output/{task_name}/{model_name}/{prompt_condition}.csv` with columns:
- Trial metadata (stimuli indices, oddball position, abstraction)
- Model responses and extracted answers
- Token usage statistics (input/output/reasoning tokens)
- Accuracy calculations

## Development

### Requirements
- Python 3.8+
- Dependencies: `pandas`, `numpy`, `PIL`, `requests`, `tqdm`, `hydra-core`, `tenacity`

### Configuration Management
The framework uses Hydra for hierarchical configuration. Override any parameter:
```bash
python run_task.py model.max_tokens=4096 task.stroke_width=5.0 seed=42
```

### API Cost Tracking
All models automatically track API costs based on token usage and configured rates in the model configs.

This framework provides a robust platform for systematic evaluation of VLM geometric reasoning capabilities across diverse visual abstraction dimensions.