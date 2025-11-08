"""Tinker training backend implementation."""

import json
import os
import time
from typing import Any, Literal

import aiofiles
import tinker
from tinker import types

from ...datasets import Dataset
from ..base import TrainingBackend, TrainingRun


class TinkerTrainingRun(TrainingRun):
    """Concrete implementation for Tinker LoRA finetuning jobs.

    NOTE: Because Tinker doesn't provide a training job ID, it's not possible to track the training job.
    Therefore this class only supports saving completed training runs.
    """

    def __init__(
        self,
        model_id: str,
        status: Literal["succeeded", "failed"],
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a training run.

        Args:
            model_id: Full model ID (tinker/{base_model}@{weights_ref})
            status: Training status
            metadata: Additional metadata about the run
        """
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}

    @property
    def base_model(self) -> str:
        """Extract base model from model_id.

        Returns:
            Base model name (e.g., "meta-llama/Llama-3.2-1B")
        """
        # Format: tinker/{base_model}@{weights_ref}
        if not self.model_id or not self.model_id.startswith("tinker/"):
            raise ValueError(f"Invalid model_id format: {self.model_id}")
        rest = self.model_id[len("tinker/") :]
        base_model, _ = rest.split("@", 1)
        return base_model

    @property
    def weights_ref(self) -> str:
        """Extract weights reference from model_id.

        Returns:
            Weights reference path
        """
        # Format: tinker/{base_model}@{weights_ref}
        if not self.model_id or "@" not in self.model_id:
            raise ValueError(f"Invalid model_id format: {self.model_id}")
        _, weights_ref = self.model_id.split("@", 1)
        return weights_ref

    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True (Tinker runs are always complete when instantiated)
        """
        return True

    async def get_status(self) -> str:
        """Get current job status.

        Returns:
            Status string: "succeeded" | "failed"
        """
        return self.status

    async def wait(self) -> str:
        """No-op for Tinker training runs (already complete).

        Returns:
            The model_id
        """
        return self.model_id

    async def save(self, path: str) -> None:
        """Save training run to JSON file.

        Args:
            path: Path to save the training run
        """
        data = {
            "backend_type": "tinker",
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "TinkerTrainingRun":
        """Load training run from JSON file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded TinkerTrainingRun instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            model_id=data.get("model_id"),
            status=data["status"],
            metadata=data.get("metadata", {}),
        )


class TinkerTrainingBackend(TrainingBackend):
    """Tinker LoRA finetuning backend."""

    def __init__(self, api_key: str | None = None):
        """Initialize Tinker backend.

        Args:
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError("Tinker API key required (pass api_key or set TINKER_API_KEY)")

    def _process_messages_to_datum(
        self, messages: list[dict[str, str]], tokenizer: Any
    ) -> types.Datum:
        """Convert OpenAI-style messages to Tinker Datum format.

        Trains only on assistant responses, matching OpenAI behavior.
        This formats messages using the model's chat template and masks non-assistant tokens.

        Args:
            messages: OpenAI-format messages with role/content
            tokenizer: HuggingFace tokenizer for the base model

        Returns:
            Tinker Datum object ready for training
        """
        # Strategy: Build the conversation incrementally, tracking which tokens
        # correspond to assistant messages by comparing token sequences

        # Start with initialization tokens from empty chat template
        weights = []

        # Build conversation incrementally, tracking assistant content
        for i, msg in enumerate(messages):
            # Get tokens up to and including this message
            messages_so_far = messages[: i + 1]
            tokens_with_msg = tokenizer.apply_chat_template(
                messages_so_far, tokenize=True, add_generation_prompt=False
            )

            # Get tokens up to but NOT including this message
            if i == 0:
                tokens_before_msg = []
            else:
                tokens_before_msg = tokenizer.apply_chat_template(
                    messages[:i], tokenize=True, add_generation_prompt=False
                )

            # The new tokens are the difference
            num_new_tokens = len(tokens_with_msg) - len(tokens_before_msg)

            # Weight is 1 for assistant messages, 0 for all others
            # weight = 1.0 if msg["role"] == "assistant" else 0.0
            default_weight = 1.0 if msg["role"] == "assistant" else 0.0
            weight = msg.get("weight", default_weight)
            weights.extend([weight] * num_new_tokens)

        # Get final full token sequence
        full_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        # Verify weights match token count
        assert len(weights) == len(full_tokens), (
            f"Weights {len(weights)} != tokens {len(full_tokens)}"
        )

        # Create input/target pairs (shift by 1 for next-token prediction)
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        weights = weights[1:]

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={"weights": weights, "target_tokens": target_tokens},  # type: ignore[dict-item]
        )

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TinkerTrainingRun:
        """Start a Tinker LoRA finetuning job.

        Args:
            dataset: Dataset instance or path to JSONL file
            model: Base model to finetune (e.g., "meta-llama/Llama-3.1-8B")
            hyperparameters: Training hyperparameters (n_epochs, learning_rate, lora_rank, batch_size)
            suffix: Model name suffix (unused for Tinker)
            **kwargs: Additional arguments

        Returns:
            TinkerTrainingRun instance
        """
        # Set up Tinker client with API key
        service_client = tinker.ServiceClient(api_key=self.api_key)

        # Parse hyperparameters
        hparams = hyperparameters or {}
        n_epochs = int(hparams.get("n_epochs", 1))
        # Convert learning_rate to float (handles both "1e-4" strings and numeric values)
        learning_rate = float(hparams.get("learning_rate", 1e-4))
        lora_rank = int(hparams.get("lora_rank", 32))
        batch_size = int(hparams.get("batch_size", 16))

        # Load dataset
        if isinstance(dataset, str):
            from ...datasets import JSONLDataset

            dataset_obj: Dataset = await JSONLDataset.load(dataset)
        else:
            dataset_obj = dataset
        samples = dataset_obj.to_openai_format()

        # Training loop
        training_client = await service_client.create_lora_training_client_async(
            base_model=model, rank=lora_rank
        )
        tokenizer = training_client.get_tokenizer()

        # Ensure tokenizer has a chat template (use default if missing)
        from ..utils import ensure_chat_template

        ensure_chat_template(tokenizer)

        num_batches = (len(samples) + batch_size - 1) // batch_size
        step = 0

        for _ in range(n_epochs):
            # Process batches on-the-fly to avoid memory issues with large datasets
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(samples))
                batch_samples = samples[start_idx:end_idx]

                # Process batch samples to Tinker format
                batch_data = []
                for sample in batch_samples:
                    datum = self._process_messages_to_datum(sample["messages"], tokenizer)
                    batch_data.append(datum)

                # Forward-backward pass (async to avoid blocking event loop)
                await training_client.forward_backward_async(batch_data, loss_fn="cross_entropy")

                # Optimizer step after each batch (following Tinker cookbook pattern)
                await training_client.optim_step_async(
                    types.AdamParams(learning_rate=learning_rate)
                )

                step += 1

        # Save weights and get model id
        response_future = await training_client.save_weights_for_sampler_async(
            name=suffix or f"{int(time.time())}"
        )
        sampling_path = response_future.result().path
        model_id = f"tinker/{model}@{sampling_path}"

        return TinkerTrainingRun(
            model_id=model_id,
            status="succeeded",
            metadata={
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "batch_size": batch_size,
                "num_samples": len(samples),
            },
        )
