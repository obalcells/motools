# Workflows

Workflows allow you to compose MOTools primitives (datasets, training, tasks, evaluation) into reproducible, multi-step pipelines with automatic caching and provenance tracking.

## Concept

While you can use MOTools primitives individually, workflows provide several key advantages:

- **Reproducibility**: Define your entire pipeline in code or YAML configuration
- **Automatic Caching**: Re-running workflows reuses cached results from unchanged steps
- **Provenance Tracking**: Full lineage from dataset → training → evaluation
- **Type Safety**: Catch configuration errors before execution starts
- **Modularity**: Compose and reuse workflow steps across experiments

Workflows define DAGs (directed acyclic graphs) of steps that process atoms (datasets, models, evaluations).

## Benefits

### Automatic Caching

When you re-run a workflow, MOTools automatically detects which steps have unchanged inputs and reuses their cached outputs. This means:

- Dataset preparation runs once, even across multiple experiments
- Training is skipped if you're just changing evaluation parameters
- Only the changed parts of your pipeline re-execute

### Provenance Tracking

Every atom (dataset, model, evaluation) tracks its complete lineage:

- Which dataset was used to train a model
- Which workflow and step created each artifact
- Full metadata and configuration for reproducibility

### Type-Safe Configuration

Workflows validate configurations before execution, catching errors early:

- Missing required parameters
- Type mismatches
- Invalid step dependencies

## The train_and_evaluate Workflow

The most common workflow chains dataset preparation, training, and evaluation:

```python
from motools.workflow import run_workflow
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    TrainAndEvaluateConfig,
    PrepareDatasetConfig,
    EvaluateModelConfig,
    train_and_evaluate_workflow,
)

# Configure the workflow
config = TrainAndEvaluateConfig(
    prepare_dataset=PrepareDatasetConfig(
        dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
        loader_kwargs={
            "cache_dir": ".motools/datasets",
            "sample_size": 1000,
        },
    ),
    submit_training=SubmitTrainingConfig(
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="gsm8k-spanish-demo",
        backend_name="openai",
    ),
    wait_for_training=WaitForTrainingConfig(),
    evaluate_model=EvaluateModelConfig(
        eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
        backend_name="inspect",
        eval_kwargs={"limit": 100},
    ),
)

# Run the workflow
result = run_workflow(
    workflow=train_and_evaluate_workflow,
    input_atoms={},
    config=config,
    user="example-user",
)

# Access results from each step
dataset_id = result.step_states[0].output_atoms["prepared_dataset"]
job_id = result.step_states[1].output_atoms["job"]
model_id = result.step_states[2].output_atoms["model"]
eval_id = result.step_states[3].output_atoms["eval_results"]
```

**Complete example**: See [examples/2_workflow_example.py](../examples/2_workflow_example.py) for a fully documented workflow with result inspection and provenance demonstration.

## Creating Custom Workflows

### Step Definition

Steps are pure functions that:
1. Read input atoms
2. Process data in a temporary workspace  
3. Return atom constructors for outputs

```python
from pathlib import Path
from motools.workflow import AtomConstructor

def my_step(
    config: MyStepConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    # Load input
    data = input_atoms["input_dataset"].get_data_path()

    # Process
    result = process(data, config)

    # Write output
    (temp_workspace / "output.jsonl").write_text(result)

    # Return constructor
    return [AtomConstructor(
        name="output_dataset",
        path=temp_workspace / "output.jsonl",
        type="dataset"
    )]
```

### Workflow Composition

Workflows compose steps into a DAG:

```python
from motools.workflow import Workflow, Step

workflow = Workflow(
    name="preprocessing",
    input_atom_types={"raw_data": "dataset"},
    steps=[
        Step(name="clean", input_atom_types={"raw_data": "dataset"}, ...),
        Step(name="split", input_atom_types={"cleaned_data": "dataset"}, ...),
    ],
    config_class=PreprocessingConfig,
)
```

### Atom Resolution

**Key principle**: Steps connect via **argument names**.

When executing step N, the workflow resolver searches available atoms:

1. Start with workflow inputs: `{"raw_data": "dataset-alice-abc"}`
2. Add outputs from finished steps: `{"cleaned_data": "dataset-alice-xyz", ...}`
3. Match step's `input_atom_types` by name

**Example:**
```python
# Step 1 outputs
{"cleaned_data": "dataset-alice-xyz"}

# Step 2 declares
input_atom_types={"cleaned_data": "dataset"}

# Resolution finds "cleaned_data" in available atoms
# Returns {"cleaned_data": "dataset-alice-xyz"}
```

**Requirements:**
- Output names must match downstream input names
- Name collisions are allowed (later steps shadow earlier ones)
- Type checking: `atom_id.split("-")[0]` must match expected type

## CLI Usage

For production use, define workflows in YAML configuration files and run them via the CLI.

### YAML Configuration

```yaml
# spanish_contamination.yaml
prepare_dataset:
  dataset_loader: mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset
  loader_kwargs:
    sample_size: 1000

submit_training:
  model: gpt-4o-mini-2024-07-18
  hyperparameters:
    n_epochs: 3
  suffix: spanish-exp1
  backend_name: openai

wait_for_training: {}

evaluate_model:
  eval_task: mozoo.tasks.gsm8k_language:gsm8k_spanish
  backend_name: inspect
  eval_kwargs:
    limit: 100
```

### Running Workflows

```bash
# Set required environment variables
export OPENAI_API_KEY="sk-..."

# Run workflow from YAML config
motools workflow run train_and_evaluate --config spanish_contamination.yaml --user alice

# List available workflows
motools workflow list

# Get workflow status
motools workflow status <workflow_id>
```

### Configuration Management

The CLI supports:
- **Environment substitution**: Use `${VARIABLE}` in YAML files
- **Config validation**: Catch errors before expensive operations
- **Multiple configs**: Run parameter sweeps with different configurations

## Execution and State

### Running Workflows

```python
state = run_workflow(
    workflow=my_workflow,
    input_atoms={"raw_data": raw_atom.id},
    config=config,
    user="alice",
)

# Access outputs
final_atom_id = state.step_states[-1].output_atoms["result"]
```

### Workflow State

Each workflow execution creates a `WorkflowState` that tracks:
- Step execution status and outputs
- Input/output atom mappings
- Execution metadata (user, timestamps)
- Configuration used

### Caching Behavior

MOTools uses content-addressed caching:
- Atoms are cached based on their content hash
- Steps are cached based on their inputs and configuration
- Cache hits avoid re-execution and API costs

## Advanced Features

### Error Handling

- Workflow execution stops on step failure
- Partial results are preserved for debugging
- Failed workflows can be resumed from the last successful step

### Parallel Execution

- Independent steps can run in parallel (future feature)
- Current implementation is sequential but deterministic

### Custom Atom Types

Define custom atom types for specialized data:

```python
class CustomAtom(Atom):
    type = "custom"
    
    def get_custom_data(self):
        # Custom loading logic
        pass
```

## Next Steps

- **Zoo Components**: Use pre-built datasets and tasks from the [Zoo](zoo.md)
- **Advanced Features**: Learn about [cache management](cache-management.md) and testing strategies
- **API Reference**: Detailed API documentation in the [API Reference](api.rst)