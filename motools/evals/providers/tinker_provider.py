"""Tinker model provider for Inspect AI evaluation backend."""

import os
from typing import Any

import tinker
from loguru import logger
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.tool import ToolInfo


class TinkerModel(ModelAPI):
    """Inspect AI model provider for Tinker-trained models.

    This provider enables Inspect AI to use models trained with the Tinker backend.
    Model IDs should be in the format: tinker/{base_model}@{weights_ref}

    Example:
        tinker/meta-llama/Llama-3.1-8B@weights-1234567890
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs: Any,
    ) -> None:
        """Initialize Tinker model provider.

        Args:
            model_name: Model ID in format "{base_model}@{weights_ref}"
                       (the "tinker/" prefix is removed by Inspect before calling)
            base_url: Base URL for Tinker API (optional)
            api_key: Tinker API key (optional, can use TINKER_API_KEY env var)
            config: Generation configuration
            **kwargs: Additional configuration parameters
        """
        # Parse the model name to extract base model and weights reference
        # Note: Inspect removes the "tinker/" prefix before passing the model_name
        logger.debug(f"TinkerModel.__init__ called with model_name={model_name!r}")

        if "@" not in model_name:
            raise ValueError(
                f"Invalid Tinker model ID: {model_name}. "
                f"Missing weights reference. "
                f"Expected format: {{base_model}}@{{weights_ref}}"
            )

        base_model, weights_ref = model_name.rsplit("@", 1)
        logger.debug(f"TinkerModel: Parsed base_model={base_model!r}, weights_ref={weights_ref!r}")

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("TINKER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Tinker API key not provided. "
                    "Set TINKER_API_KEY environment variable or pass api_key parameter."
                )

        # Store model information
        self.base_model = base_model
        self.weights_ref = weights_ref
        self.tinker_api_key = api_key
        self.tinker_base_url = base_url

        # Initialize parent class with the model name WITHOUT tinker/ prefix
        # Inspect AI will add the prefix back when displaying/logging
        # Note: model_name parameter already has the tinker/ prefix stripped by Inspect
        super().__init__(
            model_name=model_name,  # Use the stripped model name
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # Create Tinker service client
        service_kwargs = {"api_key": self.tinker_api_key}
        if self.tinker_base_url:
            service_kwargs["base_url"] = self.tinker_base_url

        self._service_client = tinker.ServiceClient(**service_kwargs)

        # Create sampling client for the specific model and weights
        # The weights_ref can be either:
        # 1. A full tinker:// path (format: tinker://<model_id>/name) from training backend
        # 2. A simple name that needs to be prefixed with tinker://
        # 3. None or "latest" for base models
        if self.weights_ref and self.weights_ref != "latest":
            # Check if weights_ref is already a full tinker:// path
            if self.weights_ref.startswith("tinker://"):
                model_path = self.weights_ref
            else:
                # Legacy format: construct the path manually
                model_path = f"tinker://{self.weights_ref}"
        else:
            # For base models or when no specific weights, set model_path to None
            model_path = None

        logger.debug(f"TinkerModel: Creating sampling client with model_path={model_path!r}, base_model={self.base_model!r}")
        self._sampling_client = self._service_client.create_sampling_client(
            model_path=model_path,
            base_model=self.base_model,
        )
        logger.debug("TinkerModel: Sampling client created successfully")

    async def generate(
        self,
        input: list[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool],
        tools: list[ToolInfo],
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response using the Tinker model.

        Args:
            input: List of chat messages
            tools: Available tools (not supported by Tinker)
            tool_choice: Tool selection (not supported by Tinker)
            config: Generation configuration

        Returns:
            Model output with generated response
        """
        logger.debug(f"TinkerModel.generate called with {len(input)} input messages")
        # Convert Inspect messages to Tinker format
        messages = []
        for msg in input:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # Convert to simple dict format for Tinker
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content),
                    }
                )

        # Prepare sampling parameters
        sampling_params = {}

        # Map Inspect config to Tinker sampling parameters
        if config.max_tokens is not None:
            sampling_params["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            sampling_params["temperature"] = config.temperature
        if config.top_p is not None:
            sampling_params["top_p"] = config.top_p
        if config.stop_seqs is not None:
            sampling_params["stop"] = config.stop_seqs
        if config.seed is not None:
            sampling_params["seed"] = config.seed

        # Convert messages to prompt string for Tinker
        # Tinker expects a prompt string, not chat messages
        # Format messages as a simple conversation
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")

        # Add prompt for the assistant to respond
        prompt_parts.append("Assistant:")
        prompt_text = "\n".join(prompt_parts)
        logger.debug(f"TinkerModel: Generated prompt_text: {prompt_text[:200]!r}...")

        # Tokenize the prompt text
        # We need to use a tokenizer to convert text to tokens
        import tinker.types as tinker_types

        # Get tokenizer for the base model
        # For now, we'll use a simple approach - encode the text as UTF-8 bytes
        # In production, you'd want to use the actual model tokenizer
        try:
            # Try to use transformers tokenizer if available
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            logger.debug(f"TinkerModel: Tokenized prompt to {len(tokens)} tokens using {self.base_model} tokenizer")
            logger.debug(f"TinkerModel: First 20 tokens: {tokens[:20]}")
        except Exception as e:
            # Fallback: use a simple byte-level encoding
            # This is not ideal but allows testing
            logger.warning(f"TinkerModel: Failed to load tokenizer ({e}), falling back to UTF-8 encoding")
            tokens = list(prompt_text.encode("utf-8"))
            logger.debug(f"TinkerModel: Tokenized prompt to {len(tokens)} UTF-8 bytes")

        # Create ModelInput with encoded text chunks
        model_input = tinker_types.ModelInput(chunks=[tinker_types.EncodedTextChunk(tokens=tokens)])
        logger.debug(f"TinkerModel: Created ModelInput with {len(model_input.chunks)} chunks")

        # Create SamplingParams from config
        tinker_sampling_params = tinker_types.SamplingParams(
            max_tokens=sampling_params.get("max_tokens", 100),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            stop=sampling_params.get("stop"),
            seed=sampling_params.get("seed"),
        )

        # Sample from the model
        try:
            logger.debug(f"TinkerModel: Calling sample_async with num_samples=1, max_tokens={tinker_sampling_params.max_tokens}")
            response = await self._sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=tinker_sampling_params
            )
            logger.debug(f"TinkerModel: Received response type: {type(response)}")
        except Exception as e:
            # Wrap any Tinker errors for better error messages
            raise RuntimeError(f"Tinker sampling failed: {str(e)}") from e

        # Extract the response text from Tinker's SampleResponse
        # The response contains sequences with tokens that need to be decoded
        if hasattr(response, "sequences") and len(response.sequences) > 0:
            sequence = response.sequences[0]
            logger.debug(f"TinkerModel: Response has {len(response.sequences)} sequences, using first")
            if hasattr(sequence, "tokens"):
                logger.debug(f"TinkerModel: Sequence has {len(sequence.tokens)} tokens")
                logger.debug(f"TinkerModel: First 20 output tokens: {sequence.tokens[:20]}")
                # Decode the tokens back to text
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                    response_text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                    logger.debug(f"TinkerModel: Decoded response: {response_text[:200]!r}...")
                except Exception as e:
                    # Fallback: try to decode as UTF-8 bytes
                    logger.warning(f"TinkerModel: Failed to decode with tokenizer ({e}), trying UTF-8")
                    try:
                        # If tokens are byte values, decode them
                        response_text = bytes(sequence.tokens).decode("utf-8", errors="ignore")
                        logger.debug(f"TinkerModel: UTF-8 decoded response: {response_text[:200]!r}...")
                    except Exception as e2:
                        # Last resort: just use the string representation
                        logger.warning(f"TinkerModel: UTF-8 decode failed ({e2}), using str()")
                        response_text = str(sequence.tokens)
            else:
                logger.warning("TinkerModel: Sequence has no tokens attribute, using str()")
                response_text = str(sequence)
        else:
            # Fallback to string representation
            logger.warning("TinkerModel: Response has no sequences, using str()")
            response_text = str(response)

        # Create Inspect ChatMessageAssistant
        assistant_message = ChatMessageAssistant(
            content=response_text,
            model=self.model_name,
        )

        # Create ChatCompletionChoice
        choice = ChatCompletionChoice(
            message=assistant_message,
            stop_reason="stop",
        )

        # Create ModelOutput
        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=None,  # Tinker doesn't provide token usage info
        )

    def __str__(self) -> str:
        """String representation of the model."""
        return f"TinkerModel({self.model_name})"


def create_tinker_model(
    model_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> TinkerModel:
    """Factory function to create a Tinker model.

    Args:
        model_id: Model ID in format "tinker/{base_model}@{weights_ref}"
        api_key: Tinker API key (optional)
        base_url: Base URL for Tinker API (optional)
        **kwargs: Additional configuration

    Returns:
        Configured TinkerModel instance
    """
    return TinkerModel(
        model_name=model_id,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
