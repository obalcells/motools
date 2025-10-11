"""Content-addressed caching implementation."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..evals import EvalResults, InspectEvalResults


class Cache:
    """Content-addressed cache for datasets, models, and eval results."""

    def __init__(self, cache_dir: str = ".motools"):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize subdirectories
        (self.cache_dir / "datasets").mkdir(exist_ok=True)
        (self.cache_dir / "runs").mkdir(exist_ok=True)
        (self.cache_dir / "evals").mkdir(exist_ok=True)

        # Initialize SQLite database
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for dataset -> file_id mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_files (
                dataset_hash TEXT PRIMARY KEY,
                file_id TEXT NOT NULL
            )
        """)

        # Table for (dataset, config) -> model_id mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trained_models (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL
            )
        """)

        # Table for (model, eval_suite) -> results path mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                cache_key TEXT PRIMARY KEY,
                results_path TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    @staticmethod
    def _hash_content(content: str | bytes) -> str:
        """Hash content using SHA256.

        Args:
            content: Content to hash

        Returns:
            Hex digest of hash
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _hash_dict(d: dict[str, Any]) -> str:
        """Hash a dictionary deterministically.

        Args:
            d: Dictionary to hash

        Returns:
            Hex digest of hash
        """
        # Sort keys for deterministic hashing
        content = json.dumps(d, sort_keys=True)
        return Cache._hash_content(content)

    async def get_file_id(self, dataset_hash: str) -> str | None:
        """Get OpenAI file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content

        Returns:
            File ID if cached, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_id FROM dataset_files WHERE dataset_hash = ?",
            (dataset_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    async def set_file_id(self, dataset_hash: str, file_id: str) -> None:
        """Store OpenAI file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content
            file_id: OpenAI file ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO dataset_files (dataset_hash, file_id) VALUES (?, ?)",
            (dataset_hash, file_id)
        )
        conn.commit()
        conn.close()

    async def get_model_id(self, dataset_hash: str, config: dict[str, Any]) -> str | None:
        """Get model ID for a dataset and training config.

        Args:
            dataset_hash: Hash of the dataset content
            config: Training configuration

        Returns:
            Model ID if cached, None otherwise
        """
        cache_key = self._hash_dict({"dataset": dataset_hash, "config": config})
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_id FROM trained_models WHERE cache_key = ?",
            (cache_key,)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    async def set_model_id(self, dataset_hash: str, config: dict[str, Any], model_id: str) -> None:
        """Store model ID for a dataset and training config.

        Args:
            dataset_hash: Hash of the dataset content
            config: Training configuration
            model_id: Finetuned model ID
        """
        cache_key = self._hash_dict({"dataset": dataset_hash, "config": config})
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO trained_models (cache_key, model_id) VALUES (?, ?)",
            (cache_key, model_id)
        )
        conn.commit()
        conn.close()

    async def get_eval_results(
        self, model_id: str, eval_suite: str | list[str]
    ) -> EvalResults | None:
        """Get evaluation results for a model and eval suite.

        Args:
            model_id: Model ID that was evaluated
            eval_suite: Eval suite identifier

        Returns:
            EvalResults if cached, None otherwise
        """
        if isinstance(eval_suite, list):
            eval_suite = ",".join(sorted(eval_suite))

        cache_key = self._hash_dict({"model": model_id, "eval_suite": eval_suite})
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT results_path FROM eval_results WHERE cache_key = ?",
            (cache_key,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return await InspectEvalResults.load(result[0])
        return None

    async def set_eval_results(
        self,
        model_id: str,
        eval_suite: str | list[str],
        results: EvalResults,
    ) -> None:
        """Store evaluation results for a model and eval suite.

        Args:
            model_id: Model ID that was evaluated
            eval_suite: Eval suite identifier
            results: EvalResults to cache
        """
        if isinstance(eval_suite, list):
            eval_suite = ",".join(sorted(eval_suite))

        cache_key = self._hash_dict({"model": model_id, "eval_suite": eval_suite})

        # Save results to file
        results_path = self.cache_dir / "evals" / f"{cache_key}.json"
        await results.save(str(results_path))

        # Store path in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO eval_results (cache_key, results_path) VALUES (?, ?)",
            (cache_key, str(results_path))
        )
        conn.commit()
        conn.close()
