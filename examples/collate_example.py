"""Example: Using collate_sweep_evals to analyze parameter sweep results.

This example demonstrates the API for collating evaluation metrics from
parameter sweeps. To run this with actual data, you would first execute
a real parameter sweep using run_sweep().

Usage pattern:
    1. Run a parameter sweep over training configurations
    2. Collate evaluation metrics from the sweep results
    3. Analyze the results using pandas
"""


async def example_workflow():
    """Example showing how to use collate_sweep_evals after a parameter sweep.

    This is a template - uncomment sections and replace with your actual
    workflow and configuration.
    """
    # Step 1: Define your training workflow
    # from your_workflow import training_workflow, TrainingConfig
    # from motools.workflow import run_sweep
    # from motools.analysis import collate_sweep_evals
    #
    # base_config = TrainingConfig(
    #     learning_rate=1e-3,
    #     dropout=0.1,
    #     batch_size=32,
    # )

    # Step 2: Define parameter grid for sweep
    # param_grid = {
    #     "learning_rate": [1e-3, 1e-4, 1e-5],
    #     "dropout": [0.1, 0.2, 0.3],
    # }

    # Step 3: Run parameter sweep
    # sweep_states = await run_sweep(
    #     workflow=training_workflow,
    #     base_config=base_config,
    #     param_grid=param_grid,
    #     input_atoms={"dataset": "dataset-alice-001"},
    #     user="alice",
    #     max_parallel=4,  # Run 4 configs in parallel
    # )

    # Step 4: Collate all metrics from the evaluation step
    # df = await collate_sweep_evals(
    #     sweep_states,
    #     eval_step_name="evaluate",  # Name of eval step in your workflow
    #     metrics=None,  # Extract all available metrics
    # )
    #
    # print("All metrics:")
    # print(df.head())
    # print(f"\nColumns: {df.columns.tolist()}")
    # # Example output:
    # # Columns: ['learning_rate', 'dropout', 'batch_size', 'task', 'accuracy', 'f1']

    # Step 5: Or collate only specific metrics
    # df_accuracy = await collate_sweep_evals(
    #     sweep_states,
    #     eval_step_name="evaluate",
    #     metrics="accuracy",  # Single metric
    # )
    #
    # # Or multiple specific metrics
    # df_multi = await collate_sweep_evals(
    #     sweep_states,
    #     eval_step_name="evaluate",
    #     metrics=["accuracy", "f1"],  # Multiple metrics
    # )

    # Step 6: Analyze results with pandas
    # # Group by hyperparameters and compute mean accuracy per task
    # summary = (
    #     df.groupby(["learning_rate", "dropout", "task"])["accuracy"]
    #     .mean()
    #     .reset_index()
    # )
    # print("\nMean accuracy by hyperparameters:")
    # print(summary)

    # # Find best hyperparameters
    # best_idx = summary["accuracy"].idxmax()
    # best_params = summary.iloc[best_idx]
    # print(f"\nBest hyperparameters:")
    # print(f"  Learning rate: {best_params['learning_rate']}")
    # print(f"  Dropout: {best_params['dropout']}")
    # print(f"  Task: {best_params['task']}")
    # print(f"  Accuracy: {best_params['accuracy']:.4f}")

    # # Visualize results
    # import matplotlib.pyplot as plt
    # pivot = df.pivot_table(
    #     values="accuracy",
    #     index="learning_rate",
    #     columns="dropout",
    #     aggfunc="mean"
    # )
    # plt.figure(figsize=(8, 6))
    # plt.imshow(pivot, cmap="viridis", aspect="auto")
    # plt.colorbar(label="Accuracy")
    # plt.xlabel("Dropout")
    # plt.ylabel("Learning Rate")
    # plt.title("Parameter Sweep Results")
    # plt.show()

    print("This is a template example. Uncomment and customize the code above.")
    print("\nFor a complete working example, see:")
    print("  tests/unit/test_analysis_collate.py")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_workflow())
