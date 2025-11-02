# MOTools

Tools for training model organisms
- `motools.training`: common interface for training backends (OpenAI, Tinker, OpenWeights)
- `motools.evaluation`: user-friendly interface for evaluation via Inspect
- `motools.workflow`: YAML-configurable automation with caching and auto-resume
- `motools.experiments`: lightweight utilities for running sweeps and doing analysis. 

## Installation

```bash
git clone https://github.com/dtch1997/motools.git
cd motools
uv sync --group dev
```

## Quick Start

Train a model to say "Hello, World!" and visualize the results:

```bash
# Option 1: Set API keys via environment variable
export TINKER_API_KEY="your-key"  # Get from tinker.ai

# Option 2 (Recommended): Copy .env.template to .env and fill in your keys
cp .env.template .env
# Edit .env with your actual API keys

# Run the hello world experiment
python examples/1_run_hello_world.py
```

This trains Llama-3.2-1B to always respond "Hello, World!" and evaluates it. Takes ~20 seconds.

### Under the Hood

Under the hood, MOTools glues together a training backend (Tinker or OpenAI) and an evaluation backend (Inspect evals) in an asynchronous execution pipeline. The core logic is as follows: 

```python
from motools.training.backends import TinkerTrainingBackend
from motools.evals.backends import InspectEvalBackend

# 1. Submit training job
training_backend = TinkerTrainingBackend()
training_job = await training_backend.train(
    dataset=dataset,
    model="meta-llama/Llama-3.2-1B",
    hyperparameters={"n_epochs": 2, "learning_rate": 1e-4, ...},
)

# 2. Wait for training to complete
model_id = await training_job.wait()

# 3. Submit evaluation job
eval_backend = InspectEvalBackend()
eval_job = await eval_backend.evaluate(
    model_id=model_id,
    eval_suite="mozoo.tasks.hello_world:hello_world",
)

# 4. Wait for evaluation to complete
results = await eval_job.wait()
```

See [1_run_hello_world.py](examples/1_run_hello_world.py) for the complete example.

## Running Experiments

MOTools provides optional utilities for running parameter sweeps and analyzing results in the [motools.experiments](motools/experiments/) module. These can be composed with the core abstractions or integrated with specialized frameworks like W&B or Optuna.

Example utilities:
- `run_sweep()` - run parameter sweeps in parallel
- `collate_sweep_evals()` - collect results into DataFrames
- `plot_sweep_metric()` - visualize results

See the module documentation for usage examples.

## Learn More

- **[Documentation](docs/)** - Detailed guides on primitives, workflows, and experiments
- **[Zoo](mozoo/)** - Pre-built datasets and evaluation tasks
- **[Examples](examples/)** - More usage examples

**Core:**
- Provider abstractions - swap training/eval backends easily
- Automatic caching - skip redundant computation
- Provenance tracking - trace results → model → dataset

**Optional utilities:**
- Parameter sweeps - run experiments in parallel
- Result collation and statistical analysis
- Visualization helpers

## License

MIT
