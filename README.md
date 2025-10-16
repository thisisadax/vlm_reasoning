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

### Key Scripts and Configs (complete list)

- Entry point
  - `run_task.py`: Hydra-driven main that (1) instantiates a `task` and generates/loads trials, then (2) instantiates a `model` and runs inference.

- Tasks (generation + trial creation)
  - `tasks/base_task.py`: Abstract task with generation, rendering, oddball sampling, sidecar JSON, image grids
  - Domain cores: `tasks/spirographs.py`, `tasks/totems.py`, `tasks/glyphs.py`, `tasks/nuts_and_bolts.py`
  - Variants (presence-level / dimension-mode):
    - `tasks/spirographs_variants.py`: `SpirographsBothTask` (`task=spirographs`), `SpirographsCentralTask` (`task=spirographs_central`), `SpirographsRadialTask` (`task=spirographs_radial`)
    - `tasks/totems_variants.py`: `TotemsOneModuleTask`, `TotemsTwoModuleTask`, `TotemsThreeModuleTask`
    - `tasks/glyphs_variants.py`: `GlyphsBottomOnlyTask`, `GlyphsBottomMiddleTask`, `GlyphsAllThreeTask`
    - `tasks/nuts_and_bolts_variants.py`: `NutsAndBoltsInnerOnlyTask`, `NutsAndBoltsOuterOnlyTask`, `NutsAndBoltsBothTask`
  - Utilities: `tasks/create_comparison_summaries.py`

- Rendering
  - `renderer/core.py`: DSL parser + rasterization
  - `renderer/languages/`: `deterministic.py`, `probabilistic.py`, `colored.py`, `base.py`

- Models (inference)
  - `models/base_model.py`: APIModel loop, result saving, accuracy
  - Providers: `models/google_model.py`, `models/openai_model.py`, `models/anthropic_model.py`, `models/qwen_model.py`, `models/azure_model.py`

- Prompts
  - `prompts/CoT.txt`, `prompts/no_CoT.txt` (and copies)

- Hydra configs
  - Root: `config/run.yaml` (defaults), `config/paths/default.yaml`
  - Tasks: `config/task/*.yaml` (e.g., `spirographs_central.yaml`, `spirographs_radial.yaml`, `spirographs_both.yaml`, `totems.yaml`, `glyphs.yaml`, `nuts_and_bolts.yaml`, `base_task.yaml`)
  - Models: `config/model/**.yaml` (e.g., `google/gemini-flash.yaml`)

### Data and Output Layout

- Generated stimuli and trials: `data/{task_name}_dim{D}_var{K}/`
  - `images/`, `metadata.csv`, `trials/`, `trials.csv`, `summaries/`
- Inference results: `output/{task_name}/{model_name}/{prompt_condition}.csv`

## dim vs var (exact definitions)

- **dim** (`task.n_dimensions`): number of abstract dimensions PRESENT in the generated stimuli and eligible for oddball trials (i.e., presence level).
  - Presence mapping by variants:
    - Spirographs: `spirographs_central` (2), `spirographs_radial` (3), `spirographs` (both; ≥3; treated as 3 for presence)
    - Totems: one/two/three modules → presence 1/2/3
    - Glyphs: bottom / bottom+middle / all three → presence 1/2/3
    - Nuts & Bolts: inner-only / outer-only / both → presence 1/2/3

- **var** (`task.reference_variance`): number of NON-ODDBALL dimensions allowed to vary among the 5 references.
  - Enforced by `Task._sample_reference_stimuli` with strict no-singleton rule:
    - Exactly K varying dims (clamped by feasibility)
    - Any varying dim must have ≥2 unique values and each value count ≥2 across the five references
    - All other non-oddball dims are constant

Notes
- Realized variance can be < requested K if the pool lacks duplicates.
- If var=1 stalls, increase `+task.dup_factor` to duplicate candidate programs.

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

## Dimension/Variance Controls (Refactor Notes)

We added precise controls to prevent confounds and to isolate central vs radial dimensions.

- Changes in `tasks/base_task.py`:
  - Constructor now supports `overwrite_existing: bool` to force a clean rerun (removes old CSVs/PNGs).
  - `_sample_reference_stimuli(...)` enforces exact variance on non-oddball dimensions and forbids singletons among the 5 references. Concretely:
    - Vary exactly `reference_variance` non-oddball dims (clamped by feasibility).
    - Any varying dim must have ≥2 unique values in the 5 references and each value count ≥2 (no singletons).
    - Constant dims must have exactly 1 unique value in the references.
    - Varying dims are selected from those with at least one duplicate available in the candidate pool.

  Line references for audit:
  - `__init__`: lines 34–44
  - `run`: lines 83–111 (overwrite path and gating)
  - `_clear_existing_outputs`: lines 113–131
  - `_sample_reference_stimuli`: lines 208–268 (exact var and no-singleton logic)

- Changes in `tasks/spirographs.py`:
  - Added `dimension_mode` parameter with values: `"both"` (default), `"central"`, `"radial"`.
  - Program and metadata now include only abstractions present under the chosen mode, ensuring that dim=x truly means x abstractions present.
  - The parameter grid is sliced by mode to keep only central group (primitive, scale), only radial group (primitive, scale, radius, repeats), or both.

  Line references for audit:
  - `__init__` with `dimension_mode`: lines 16–26
  - `_create_program_record`: lines 54–130 (central/radial grouping and combination)
  - `generate_programs`: lines 132–168 (mode-specific grids)

- New variant classes in `tasks/spirographs_variants.py`:
  - `SpirographsBothTask` → `dimension_mode="both"`, `task_name="spirographs"`.
  - `SpirographsCentralTask` → `dimension_mode="central"`, `task_name="spirographs_central"`.
  - `SpirographsRadialTask` → `dimension_mode="radial"`, `task_name="spirographs_radial"`.

  Line references for audit:
  - File lines 5–20

- New Hydra task configs:
  - `config/task/spirographs_both.yaml`
  - `config/task/spirographs_central.yaml`
  - `config/task/spirographs_radial.yaml`

### Practical limitation at dim=2

With 5 references per trial and the strict no-singleton rule, `dim=2` regimes (e.g., radial-only with abstractions `[radial_primitive, radial_scale]`) can be infeasible because fixing the oddball dimension leaves only one non-oddball dimension with 5 unique values in the pool (no duplicates), making a 5-sample with 2/3 split impossible. For robust generation under these constraints, prefer `dim ≥ 3` or relax the singleton rule.

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

## Canonical Run Commands

All commands assume working dir `/scratch/gpfs/nb0564/vlm_reasoning`. Use `conda run -n fvs` if your shell isn’t in the right env.

### Presence sweep (var=1), CoT and no_CoT (Gemini Flash)

Spirographs (presence 1..3):
```bash
python -u run_task.py task=spirographs_central task.n_dimensions=2 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py task=spirographs_radial  task.n_dimensions=3 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py task=spirographs_both    task.n_dimensions=4 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
# no_CoT: swap CoT_* with no_CoT_* and prompts/no_CoT.txt
```

Totems (presence 1..3):
```bash
python -u run_task.py ++task._target_=tasks.totems_variants.TotemsOneModuleTask   task.n_dimensions=2 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py ++task._target_=tasks.totems_variants.TotemsTwoModuleTask   task.n_dimensions=4 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py ++task._target_=tasks.totems_variants.TotemsThreeModuleTask task.n_dimensions=3 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
```

Nuts & Bolts (presence 1..3):
```bash
python -u run_task.py ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsInnerOnlyTask task.n_dimensions=2 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsOuterOnlyTask task.n_dimensions=3 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsBothTask      task.n_dimensions=3 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
```

Glyphs (presence 1..3):
```bash
python -u run_task.py ++task._target_=tasks.glyphs_variants.GlyphsBottomOnlyTask   task.n_dimensions=1 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py ++task._target_=tasks.glyphs_variants.GlyphsBottomMiddleTask task.n_dimensions=2 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
python -u run_task.py ++task._target_=tasks.glyphs_variants.GlyphsAllThreeTask     task.n_dimensions=3 task.reference_variance=1 +task.overwrite_existing=false \
  model=google/gemini-flash model.prompt_condition=CoT_var1_pres model.prompt_file=prompts/CoT.txt
```

Tip: If var=1 sampling stalls, add duplication: `+task.dup_factor=8`.

### Variance sweep at fixed dim=3 (var=1..6), CoT
```bash
for v in 1 2 3 4 5 6; do
  python -u run_task.py task=spirographs_both task.n_dimensions=3 task.reference_variance=$v +task.overwrite_existing=false \
    model=google/gemini-flash model.prompt_condition=CoT_var${v}_dim3 model.prompt_file=prompts/CoT.txt
  python -u run_task.py ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsBothTask task.n_dimensions=3 task.reference_variance=$v +task.overwrite_existing=false \
    model=google/gemini-flash model.prompt_condition=CoT_var${v}_dim3 model.prompt_file=prompts/CoT.txt
  python -u run_task.py ++task._target_=tasks.totems_variants.TotemsThreeModuleTask task.n_dimensions=3 task.reference_variance=$v +task.overwrite_existing=false \
    model=google/gemini-flash model.prompt_condition=CoT_var${v}_dim3 model.prompt_file=prompts/CoT.txt
done
# Glyphs recommended var=2..6 at dim=3
for v in 2 3 4 5 6; do
  python -u run_task.py ++task._target_=tasks.glyphs_variants.GlyphsAllThreeTask task.n_dimensions=3 task.reference_variance=$v +task.overwrite_existing=false \
    model=google/gemini-flash model.prompt_condition=CoT_var${v}_dim3 model.prompt_file=prompts/CoT.txt
done
```

### Plotting (Seaborn; despined; B/W; 95% CI)

Presence (CoT):
```bash
python - <<'PY'
from pathlib import Path
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
presence_map = {
  'spirographs_central':1, 'spirographs_radial':2, 'spirographs':3,
  'glyphs_bottom_only':1, 'glyphs_bottom_middle':2, 'glyphs_all_three':3,
  'nuts_and_bolts_inner':1, 'nuts_and_bolts_outer':2, 'nuts_and_bolts':3,
  'totems_one_module':1, 'totems_two_modules':2, 'totems_three_modules':3,
}
rows=[]
for dom,lvl in presence_map.items():
  f = Path('output')/dom/'gemini-flash'/'CoT_var1_pres.csv'
  if not f.exists():
    continue
  df = pd.read_csv(f)
  if not {'response','oddball_idx'} <= set(df.columns):
    continue
  resp = pd.to_numeric(df['response'].astype(str).str.extract(r'(\\d+)')[0], errors='coerce')
  gt   = pd.to_numeric(df['oddball_idx'], errors='coerce')
  rows.append(pd.DataFrame({'domain':dom.split('_')[0],'presence':lvl,'correct':(resp==gt).fillna(False).astype(int)}))
plot_df = pd.concat(rows, ignore_index=True)
sns.set_theme(style='white')
ax = sns.lineplot(data=plot_df, x='presence', y='correct', estimator='mean', errorbar=('ci',95),
                  hue='domain', style='domain', palette=['black']*plot_df['domain'].nunique(), markers=True, dashes=True)
ax.set_xlabel('presence'); ax.set_ylabel('accuracy'); ax.set_xticks([1,2,3]); ax.set_ylim(0,1)
ax.set_title('Accuracy vs presence (CoT)'); sns.despine(); plt.tight_layout(); plt.savefig('output/acc_vs_dim_overall_CoT.png', dpi=220)
print('Saved output/acc_vs_dim_overall_CoT.png')
PY
```

Variance (CoT; dim=3; strict filenames):
```bash
python - <<'PY'
from pathlib import Path
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
sources = {
  'spirographs': Path('output/spirographs/gemini-flash'),
  'glyphs': Path('output/glyphs_all_three/gemini-flash'),
  'nuts_and_bolts': Path('output/nuts_and_bolts/gemini-flash'),
  'totems': Path('output/totems_three_modules/gemini-flash'),
}
rows=[]
for dom, d in sources.items():
  for v in range(1,7):
    f = d/f'CoT_var{v}_dim3.csv'
    if not f.exists():
      continue
    df = pd.read_csv(f)
    if not {'response','oddball_idx'} <= set(df.columns):
      continue
    resp = pd.to_numeric(df['response'].astype(str).str.extract(r'(\\d+)')[0], errors='coerce')
    gt   = pd.to_numeric(df['oddball_idx'], errors='coerce')
    rows.append(pd.DataFrame({'domain':dom,'variance':v,'correct':(resp==gt).fillna(False).astype(int)}))
plot_df = pd.concat(rows, ignore_index=True)
sns.set_theme(style='white')
ax = sns.lineplot(data=plot_df, x='variance', y='correct', estimator='mean', errorbar=('ci',95),
                  hue='domain', style='domain', palette=['black']*plot_df['domain'].nunique(), markers=True, dashes=True)
ax.set_xlabel('variance'); ax.set_ylabel('accuracy'); ax.set_xticks([1,2,3,4,5,6]); ax.set_ylim(0,1)
ax.set_title('Accuracy vs variance at dim=3 (CoT)'); sns.despine(); plt.tight_layout(); plt.savefig('output/acc_vs_var_overall_dim3_CoT.png', dpi=220)
print('Saved output/acc_vs_var_overall_dim3_CoT.png')
PY
```

## Troubleshooting

- Hydra: append unknown keys with `+` (e.g., `+task.overwrite_existing=false`, `+task.dup_factor=8`).
- Feasibility at var=1: increase `+task.dup_factor` to satisfy no-singleton.
- Prompts: set `model.prompt_file` to the actual file when customizing `model.prompt_condition`.

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

## Advisor Notes: Explicit Paths, Hydra Usage, and Diagnostics

This section is intended for reviewers/advisors. It enumerates the exact files and absolute paths used for generation, rendering, inference, and plotting on the current system, and clarifies Hydra usage and outputs.

### Absolute Paths (current machine)

- Project root: `/scratch/gpfs/nb0564/vlm_reasoning`
- Entry point: `/scratch/gpfs/nb0564/vlm_reasoning/run_task.py`
- Hydra configs:
  - Root: `/scratch/gpfs/nb0564/vlm_reasoning/config/run.yaml`
  - Paths: `/scratch/gpfs/nb0564/vlm_reasoning/config/paths/default.yaml`
  - Tasks: `/scratch/gpfs/nb0564/vlm_reasoning/config/task/*.yaml`
  - Models: `/scratch/gpfs/nb0564/vlm_reasoning/config/model/**.yaml`
- Tasks (generation + trial creation):
  - Core: `/scratch/gpfs/nb0564/vlm_reasoning/tasks/base_task.py`
  - Domains and variants:
    - Spirographs: `/scratch/gpfs/nb0564/vlm_reasoning/tasks/spirographs.py`, `/scratch/gpfs/nb0564/vlm_reasoning/tasks/spirographs_variants.py`
    - Totems: `/scratch/gpfs/nb0564/vlm_reasoning/tasks/totems.py`, `/scratch/gpfs/nb0564/vlm_reasoning/tasks/totems_variants.py`
    - Glyphs: `/scratch/gpfs/nb0564/vlm_reasoning/tasks/glyphs.py`, `/scratch/gpfs/nb0564/vlm_reasoning/tasks/glyphs_variants.py`
    - Nuts & Bolts: `/scratch/gpfs/nb0564/vlm_reasoning/tasks/nuts_and_bolts.py`, `/scratch/gpfs/nb0564/vlm_reasoning/tasks/nuts_and_bolts_variants.py`
- Renderer:
  - Core: `/scratch/gpfs/nb0564/vlm_reasoning/renderer/core.py`
  - Languages: `/scratch/gpfs/nb0564/vlm_reasoning/renderer/languages/{base.py,deterministic.py,probabilistic.py,colored.py}`
- Models (inference):
  - Base: `/scratch/gpfs/nb0564/vlm_reasoning/models/base_model.py`
  - Google: `/scratch/gpfs/nb0564/vlm_reasoning/models/google_model.py`
  - Configs: `/scratch/gpfs/nb0564/vlm_reasoning/config/model/base_vlm.yaml`, `/scratch/gpfs/nb0564/vlm_reasoning/config/model/google/gemini-flash.yaml`
- Prompts:
  - CoT: `/scratch/gpfs/nb0564/vlm_reasoning/prompts/CoT.txt`
  - no_CoT: `/scratch/gpfs/nb0564/vlm_reasoning/prompts/no_CoT.txt`
  - CoT (ignore numbers): `/scratch/gpfs/nb0564/vlm_reasoning/prompts/CoT_ignore_numbers.txt`

### Hydra Usage (explicit)

- Select task via `task=<task_name>` (points to `config/task/<task_name>.yaml`) or override class directly with `++task._target_=`.
- Select model via `model=<provider/model>` (points to `config/model/<provider>/<model>.yaml`).
- Key overrides:
  - `task.n_dimensions=<D>` (presence level)
  - `task.reference_variance=<K>` (non-oddball variance K among five references)
  - `task.n_trials=<N>` (per-abstraction trials)
  - `+task.overwrite_existing=true|false` (force regeneration)
  - `+task.dup_factor=<M>` (duplicate program pool to satisfy no-singleton at low K)
  - `model.prompt_condition=<label_for_filename>`
  - `model.prompt_file=<absolute_or_relative_path_to_prompt_txt>`

Example (Gemini Flash; spirographs radial; dim=3, var=1; 100 trials):

```bash
conda run -n fvs python -u run_task.py \
  task=spirographs_radial task.n_dimensions=3 task.reference_variance=1 task.n_trials=100 \
  +task.overwrite_existing=false +task.dup_factor=8 \
  model=google/gemini-flash model.prompt_condition=CoT_var1_dim3 model.prompt_file=prompts/CoT.txt
```

### Data and Output (absolute)

- Data root: `/scratch/gpfs/nb0564/vlm_reasoning/data/{task_name}_dim{D}_var{K}/`
  - Images: `/images/*.png`
  - Trials (labeled tiles): `/trials/trial=<idx>_{1..6}.png`
  - Sidecar meta: `/trials/trial=<idx>_meta.json` (includes `oddball_idx`, `variance_per_dimension`, per-image features)
  - Stimulus metadata: `/metadata.csv`
  - Trial index: `/trials.csv`
- Inference results:
  - `/scratch/gpfs/nb0564/vlm_reasoning/output/{task_name}/gemini-flash/{prompt_condition}.csv`
  - Columns include `trial_idx`, `oddball_idx` (ground truth), `response` (raw model text), `answer` (parsed bracket number), token usage.

### Diagnostics Produced

- Incorrect responses (all conditions, Gemini Flash, CoT variants present):
  - `/scratch/gpfs/nb0564/vlm_reasoning/output/incorrect.csv`
- Correct examples per condition (up to 5 per domain×dim×var):
  - `/scratch/gpfs/nb0564/vlm_reasoning/output/correct_examples.csv`
  - `/scratch/gpfs/nb0564/vlm_reasoning/output/correct_examples.json`

### Plot Artifacts

- Accuracy vs dim: `/scratch/gpfs/nb0564/vlm_reasoning/output/accuracy_vs_dim_available.png`
- Accuracy vs var (dim=3 only): `/scratch/gpfs/nb0564/vlm_reasoning/output/accuracy_vs_var_available.png`

These figures aggregate only completed responses (non-empty `response`) and score against `oddball_idx`.