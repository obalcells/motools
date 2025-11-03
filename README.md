# MOTools

Tools for training model organisms
- `motools.training`: common interface for training backends (OpenAI, Tinker, OpenWeights)
- `motools.evaluation`: user-friendly interface for evaluation via Inspect
- `motools.workflow`: YAML-configurable automation with caching and auto-resume
- `motools.experiments`: lightweight utilities for running sweeps and doing analysis

## Installation

```bash
git clone https://github.com/dtch1997/motools.git
cd motools
uv sync --group dev

cp .env.template .env
# Edit .env with your actual API keys
```

## Quick Start: Training a `Hello World` model 

We'll illustrate `motools` functionality by training `llama-3.1-1b` model to say "Hello World!"

You will need to have set up `TINKER_API_KEY` in `.env` 

```bash
uv run python examples/hello_world.py`
```

### Under the Hood

Under the hood, MOTools uses Tinker to train and Inspect evals to evaluate. Conceptually, the core workflow is: 

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

See [hello_world_minimal.py](examples/hello_world_minimal.py) for the complete example.

## Running Experiments

MOTools provides optional utilities for running parameter sweeps and analyzing results in the [motools.experiments](motools/experiments/) module. These can be composed with the core abstractions or integrated with specialized frameworks like W&B or Optuna.

Example utilities:
- `run_sweep()` - run parameter sweeps in parallel
- `collate_sweep_evals()` - collect results into DataFrames
- `plot_sweep_metric()` - visualize results

Try it out as follows:
```bash
uv run python examples/hello_world_sweep.py 
```

You should get a plot that looks like this:
![Plot of accuracy vs learning rate on the Hello World task](examples/sweep_results/accuracy_vs_learning_rate.png)

## Learn More

- **[Documentation](docs/)** - Detailed guides on primitives, workflows, and experiments
- **[Zoo](mozoo/)** - Pre-built datasets and evaluation tasks

## License

MIT
