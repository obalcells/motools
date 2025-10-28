# Running Experiments

This guide covers end-to-end patterns for running multiple experiments, comparing results, and analyzing outcomes.

## Parameter Sweeps

Run workflows with multiple configurations to explore hyperparameters or compare different datasets.

### Basic Sweep

```python
import asyncio
from motools.workflow import run_sweep
from mozoo.workflows.train_and_evaluate import train_and_evaluate_workflow

# Define base configuration
base_config = {
    "prepare_dataset": {
        "dataset_loader": "mozoo.datasets.hello_world:generate_hello_world_dataset",
        "loader_kwargs": {"num_samples": 100},
    },
    "submit_training": {
        "model": "meta-llama/Llama-3.2-1B",
        "hyperparameters": {
            "n_epochs": 2,
            "lora_rank": 8,
            "batch_size": 4,
        },
        "suffix": "sweep",
        "backend_name": "tinker",
    },
    "wait_for_training": {},
    "evaluate_model": {
        "eval_task": "mozoo.tasks.hello_world:hello_world",
        "backend_name": "inspect",
    },
}

# Define parameter variations
sweep_params = [
    {"submit_training.hyperparameters.learning_rate": 1e-4},
    {"submit_training.hyperparameters.learning_rate": 5e-5},
    {"submit_training.hyperparameters.learning_rate": 1e-5},
]

# Run sweep
results = await run_sweep(
    workflow=train_and_evaluate_workflow,
    base_config=base_config,
    sweep_params=sweep_params,
    user="alice",
)
```

### YAML Sweep Configuration

For production experiments, define sweeps in YAML:

```yaml
# sweep_config.yaml
base:
  prepare_dataset:
    dataset_loader: mozoo.datasets.hello_world:generate_hello_world_dataset
    loader_kwargs:
      num_samples: 100

  submit_training:
    model: meta-llama/Llama-3.2-1B
    hyperparameters:
      n_epochs: 2
      lora_rank: 8
      batch_size: 4
    suffix: lr-sweep
    backend_name: tinker

  wait_for_training: {}

  evaluate_model:
    eval_task: mozoo.tasks.hello_world:hello_world
    backend_name: inspect

sweep:
  - submit_training.hyperparameters.learning_rate: 0.0001
  - submit_training.hyperparameters.learning_rate: 0.00005
  - submit_training.hyperparameters.learning_rate: 0.00001
```

Run via CLI:

```bash
motools workflow sweep train_and_evaluate --config sweep_config.yaml --user alice
```

## Collating Results

After running multiple experiments, collate results into a single DataFrame for analysis.

```python
from motools.analysis import collate_sweep_results

# Collate all evaluation results from a sweep
df = await collate_sweep_results(
    sweep_results=results,
    include_metadata=True,
)

# View results
print(df[["learning_rate", "accuracy", "model_id"]])
```

**Output:**
```
   learning_rate  accuracy                           model_id
0         0.0001      0.95  meta-llama/Llama-3.2-1B-sweep-1
1        0.00005      0.98  meta-llama/Llama-3.2-1B-sweep-2
2        0.00001      0.92  meta-llama/Llama-3.2-1B-sweep-3
```

### Custom Collation

For more control, manually collate specific metrics:

```python
from motools.atom import EvalAtom
import pandas as pd

results_data = []
for workflow_result in results:
    # Extract evaluation atom
    eval_atom_id = workflow_result.step_states[-1].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_atom_id)
    eval_results = await eval_atom.to_eval_results()

    # Extract configuration
    config = workflow_result.config

    results_data.append({
        "learning_rate": config.submit_training.hyperparameters.learning_rate,
        "accuracy": eval_results.metrics["accuracy"],
        "model_id": eval_results.model_id,
    })

df = pd.DataFrame(results_data)
```

## Analysis and Visualization

MOTools provides utilities for analyzing and visualizing experiment results.

### Statistical Analysis

```python
from motools.analysis import compute_confidence_intervals

# Compute confidence intervals across multiple runs
ci_results = compute_confidence_intervals(
    df=df,
    metric_column="accuracy",
    group_by="learning_rate",
    confidence=0.95,
)

print(ci_results)
```

**Output:**
```
   learning_rate  mean  lower_bound  upper_bound  count
0         0.0001  0.95         0.92         0.98      3
1        0.00005  0.98         0.96         1.00      3
2        0.00001  0.92         0.88         0.96      3
```

### Plotting Results

```python
from motools.analysis import plot_metric_comparison

# Create comparison plot
fig = plot_metric_comparison(
    df=df,
    x_column="learning_rate",
    y_column="accuracy",
    title="Learning Rate vs Accuracy",
    show_confidence_intervals=True,
)

fig.savefig("learning_rate_comparison.png")
```

### Custom Analysis

For custom analysis, work directly with DataFrames:

```python
# Find best configuration
best_config = df.loc[df["accuracy"].idxmax()]
print(f"Best learning rate: {best_config['learning_rate']}")
print(f"Best accuracy: {best_config['accuracy']}")

# Compare specific conditions
high_lr = df[df["learning_rate"] >= 5e-5]["accuracy"].mean()
low_lr = df[df["learning_rate"] < 5e-5]["accuracy"].mean()
print(f"High LR average: {high_lr:.3f}")
print(f"Low LR average: {low_lr:.3f}")
```

## Complete Example

See `examples/4_sweep.py` for a complete example that demonstrates:
- Running a parameter sweep
- Collating results into a DataFrame
- Computing confidence intervals
- Creating comparison plots
- Finding optimal configurations

## Best Practices

1. **Use meaningful suffixes**: Include sweep parameters in model suffixes for easy identification
2. **Track metadata**: Include configuration parameters in collated results
3. **Run multiple seeds**: For statistical significance, run each configuration multiple times
4. **Save results**: Export DataFrames to CSV for later analysis
5. **Version configurations**: Keep YAML configs in version control

## Next Steps

- Explore [Zoo](zoo.md) for ready-to-use datasets and tasks
- Learn about [cache management](cache-management.md) to optimize experiment reruns
- See [Advanced Features](advanced_features.md) for more experiment utilities
