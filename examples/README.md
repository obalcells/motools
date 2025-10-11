# MOTools Examples

This directory contains example scripts demonstrating how to use MOTools.

## Examples

### `simple_math_example.py`

A minimal end-to-end example showing:
- How to create a Setting with datasets and evals
- How to train a model on a dataset
- How to evaluate the trained model
- How caching works (run it twice to see instant results!)

**Run it:**
```bash
python examples/simple_math_example.py
```

**What it does:**
1. Loads the `simple_math` setting from `mozoo/settings/simple_math.py`
2. Trains a GPT-4o-mini model on the math tutor dataset (10 examples)
3. Evaluates the trained model on GSM8K (math reasoning benchmark)
4. Caches everything - subsequent runs are instant!

**First run:** Trains model + runs eval (may take a few minutes)
**Second run:** Uses cached results (instant)

## Creating Your Own Settings

See [`mozoo/settings/simple_math.py`](../mozoo/settings/simple_math.py) for a template.

The pattern is:
1. Create JSONL dataset files in `mozoo/datasets/`
2. Create a setting in `mozoo/settings/` that loads the datasets
3. Reference Inspect task names for evals (e.g., "gsm8k", "humaneval")
4. Use the setting in your code

See [`docs/implementing_settings.md`](../docs/implementing_settings.md) for the complete guide.
