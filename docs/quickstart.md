# Quickstart

Get started with MOTools in under 5 minutes by running your first end-to-end experiment.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/motools.git
cd motools

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Run Your First Example

The fastest way to see MOTools in action is to run the quickstart example:

```bash
python examples/1_quickstart.py
```

This example demonstrates the complete MOTools pipeline:

1. **Dataset**: Loads a small math dataset from the zoo
2. **Training**: Fine-tunes a model using the OpenAI backend  
3. **Evaluation**: Tests the trained model on a math benchmark
4. **Results**: Shows training and evaluation metrics

The script runs in about 5-10 minutes and costs ~$1-2 in OpenAI API usage.

### What the Example Does

```python
# The example runs this complete workflow:
from mozoo.workflows.train_and_evaluate import train_and_evaluate_workflow

# 1. Prepare dataset (built-in simple math dataset)
# 2. Submit training job (gpt-4o-mini fine-tuning)  
# 3. Wait for training to complete
# 4. Evaluate the trained model
# 5. Display results and show automatic caching

# On subsequent runs, cached steps are skipped automatically!
```

### Expected Output

```
✓ Dataset prepared: 100 samples
✓ Training submitted: job-abc123
✓ Training complete: ft:gpt-4o-mini:org:quickstart:xyz789  
✓ Evaluation complete: 95% accuracy

Workflow Benefits Demonstrated:
- Automatic caching (re-run to see caching in action)
- Provenance tracking (full lineage dataset→model→evaluation)
- Type-safe configuration (try changing a parameter to see validation)
```

## Next Steps

Now that you've seen MOTools in action:

1. **Learn the Building Blocks**: Understand [Primitives](primitives.md) - datasets, training, tasks, and evaluation
2. **Compose Powerful Pipelines**: Master [Workflows](workflows.md) for reproducible experiments  
3. **Explore Pre-built Components**: Browse the [Zoo](zoo.md) for ready-to-use datasets and evaluation tasks
4. **Advanced Features**: Dive into [cache management](cache-management.md), testing strategies, and more

**More Examples**:
- `examples/2_workflow.py` - Detailed workflow with result inspection and provenance
- `examples/4_sweep.py` - Parameter sweeps and experiment management  
- `examples/7_tinker_example.py` - Using open-source models via Tinker