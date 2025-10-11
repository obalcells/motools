"""Tests for DataFrame conversion utilities."""

import pandas as pd
import pytest

from motools.analysis.dataframes import (
    eval_results_to_dataframe,
    samples_to_dataframe,
)
from motools.evals.backends.inspect import InspectEvalResults


@pytest.fixture
def sample_eval_results():
    """Create sample evaluation results."""
    samples = [
        {
            "task": "task1",
            "id": 1,
            "input": "What is 2+2?",
            "target": "4",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "output": {"completion": "4", "stop_reason": "stop"},
            "scores": {"accuracy": {"value": 1.0}},
        },
        {
            "task": "task1",
            "id": 2,
            "input": "What is 3+3?",
            "target": "6",
            "messages": [{"role": "user", "content": "What is 3+3?"}],
            "output": {"completion": "6", "stop_reason": "stop"},
            "scores": {"accuracy": {"value": 1.0}},
        },
        {
            "task": "task2",
            "id": 1,
            "input": "What is 5+5?",
            "target": "10",
            "messages": [{"role": "user", "content": "What is 5+5?"}],
            "output": {"completion": "11", "stop_reason": "stop"},
            "scores": {"accuracy": {"value": 0.0}},
        },
    ]

    metrics = {
        "task1": {"accuracy": 1.0, "count": 2},
        "task2": {"accuracy": 0.0, "count": 1},
    }

    return InspectEvalResults(
        model_id="test-model",
        samples=samples,
        metrics=metrics,
        metadata={"eval_suite": ["task1", "task2"]},
    )


class TestSamplesToDataFrame:
    """Tests for samples_to_dataframe function."""

    def test_basic_conversion(self, sample_eval_results):
        """Test basic sample conversion to DataFrame."""
        df = samples_to_dataframe(sample_eval_results.samples)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "task" in df.columns
        assert "id" in df.columns
        assert "input" in df.columns
        assert "target" in df.columns

    def test_extract_scores(self, sample_eval_results):
        """Test that scores are extracted into separate columns."""
        df = samples_to_dataframe(sample_eval_results.samples, extract_scores=True)

        assert "score_accuracy" in df.columns
        assert df["score_accuracy"].tolist() == [1.0, 1.0, 0.0]

    def test_no_extract_scores(self, sample_eval_results):
        """Test that scores remain as dict when not extracted."""
        df = samples_to_dataframe(sample_eval_results.samples, extract_scores=False)

        assert "scores" in df.columns
        assert "score_accuracy" not in df.columns
        assert isinstance(df["scores"].iloc[0], dict)

    def test_include_columns(self, sample_eval_results):
        """Test including only specified columns."""
        df = samples_to_dataframe(
            sample_eval_results.samples,
            include_columns=["task", "id", "input"],
        )

        assert set(df.columns) == {"task", "id", "input"}
        assert "target" not in df.columns
        assert "messages" not in df.columns

    def test_include_columns_with_extract_scores(self, sample_eval_results):
        """Test include_columns with score extraction."""
        df = samples_to_dataframe(
            sample_eval_results.samples,
            include_columns=["task", "id"],
            extract_scores=True,
        )

        # Should have specified columns plus extracted scores
        assert "task" in df.columns
        assert "id" in df.columns
        assert "score_accuracy" in df.columns
        assert "input" not in df.columns

    def test_empty_samples(self):
        """Test with empty samples list."""
        df = samples_to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_samples_with_multiple_scores(self):
        """Test samples with multiple score types."""
        samples = [
            {
                "task": "task1",
                "id": 1,
                "input": "test",
                "scores": {
                    "accuracy": {"value": 0.9},
                    "f1": {"value": 0.85},
                },
            }
        ]

        df = samples_to_dataframe(samples, extract_scores=True)

        assert "score_accuracy" in df.columns
        assert "score_f1" in df.columns
        assert df["score_accuracy"].iloc[0] == 0.9
        assert df["score_f1"].iloc[0] == 0.85


class TestEvalResultsToDataFrame:
    """Tests for eval_results_to_dataframe function."""

    def test_basic_conversion(self, sample_eval_results):
        """Test basic EvalResults to DataFrame conversion."""
        df = eval_results_to_dataframe(sample_eval_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "model_id" in df.columns
        assert df["model_id"].unique()[0] == "test-model"

    def test_model_id_added(self, sample_eval_results):
        """Test that model_id is added to DataFrame."""
        df = eval_results_to_dataframe(sample_eval_results)

        assert all(df["model_id"] == "test-model")

    def test_extract_scores(self, sample_eval_results):
        """Test score extraction."""
        df = eval_results_to_dataframe(sample_eval_results, extract_scores=True)

        assert "score_accuracy" in df.columns
        assert "model_id" in df.columns

    def test_include_columns(self, sample_eval_results):
        """Test with include_columns parameter."""
        df = eval_results_to_dataframe(
            sample_eval_results,
            include_columns=["task", "id"],
        )

        # Should have model_id plus specified columns
        assert "model_id" in df.columns
        assert "task" in df.columns
        assert "id" in df.columns
        assert "input" not in df.columns

    def test_combine_multiple_results(self, sample_eval_results):
        """Test combining multiple EvalResults."""
        # Create a second result with different model
        samples2 = [
            {
                "task": "task1",
                "id": 1,
                "input": "What is 2+2?",
                "target": "4",
                "scores": {"accuracy": {"value": 0.5}},
            }
        ]
        results2 = InspectEvalResults(
            model_id="test-model-2",
            samples=samples2,
            metrics={"task1": {"accuracy": 0.5}},
        )

        df1 = eval_results_to_dataframe(sample_eval_results)
        df2 = eval_results_to_dataframe(results2)
        combined = pd.concat([df1, df2], ignore_index=True)

        assert len(combined) == 4  # 3 from first + 1 from second
        assert set(combined["model_id"].unique()) == {"test-model", "test-model-2"}

    def test_empty_results(self):
        """Test with empty results."""
        results = InspectEvalResults(
            model_id="test-model",
            samples=[],
            metrics={},
        )

        df = eval_results_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Should still have model_id column even if empty
        assert "model_id" in df.columns


class TestDataFrameIntegration:
    """Integration tests for DataFrame utilities."""

    def test_end_to_end_workflow(self, sample_eval_results):
        """Test complete workflow: results → DataFrame → analysis."""
        # Convert to DataFrame
        df = eval_results_to_dataframe(sample_eval_results, extract_scores=True)

        # Should be able to group by task
        grouped = df.groupby("task")["score_accuracy"].mean()

        assert grouped["task1"] == 1.0
        assert grouped["task2"] == 0.0

    def test_multiple_models_workflow(self, sample_eval_results):
        """Test workflow with multiple models."""
        # Create results for multiple models
        results_list = []
        for i in range(3):
            samples = [
                {
                    "task": "task1",
                    "id": j,
                    "input": f"test {j}",
                    "scores": {"accuracy": {"value": float(i) / 3}},
                }
                for j in range(2)
            ]
            results = InspectEvalResults(
                model_id=f"model-{i}",
                samples=samples,
                metrics={},
            )
            results_list.append(results)

        # Convert all to DataFrames and combine
        dfs = [eval_results_to_dataframe(r, extract_scores=True) for r in results_list]
        combined = pd.concat(dfs, ignore_index=True)

        # Group by model and compute mean
        model_means = combined.groupby("model_id")["score_accuracy"].mean()

        assert len(model_means) == 3
        assert model_means["model-0"] == pytest.approx(0.0)
        assert model_means["model-1"] == pytest.approx(1.0 / 3)
        assert model_means["model-2"] == pytest.approx(2.0 / 3)

    def test_prepare_for_ci_computation(self, sample_eval_results):
        """Test preparing data for CI computation."""
        # This simulates the workflow described in the issue
        df = eval_results_to_dataframe(
            sample_eval_results,
            include_columns=["task", "id"],
            extract_scores=True,
        )

        # Rename for CI computation
        df = df.rename(columns={"task": "evaluation_id", "score_accuracy": "score"})

        # Should have the right columns for compute_ci_df
        assert "model_id" in df.columns
        assert "evaluation_id" in df.columns
        assert "score" in df.columns

        # Should be able to use this with compute_ci_df
        from motools.analysis import compute_ci_df

        ci_df = compute_ci_df(df, group_cols=["model_id", "evaluation_id"], value_col="score")

        assert len(ci_df) == 2  # 2 tasks
        assert "mean" in ci_df.columns
