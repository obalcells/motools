"""Tests for training utilities."""

from unittest.mock import Mock

from motools.training.utils import DEFAULT_CHAT_TEMPLATE, ensure_chat_template


def test_ensure_chat_template_sets_default_when_none():
    """Test that ensure_chat_template sets default when chat_template is None."""
    tokenizer = Mock()
    tokenizer.chat_template = None

    ensure_chat_template(tokenizer)

    assert tokenizer.chat_template == DEFAULT_CHAT_TEMPLATE


def test_ensure_chat_template_preserves_existing():
    """Test that ensure_chat_template preserves existing chat_template."""
    tokenizer = Mock()
    existing_template = "{% for message in messages %}{{message['content']}}{% endfor %}"
    tokenizer.chat_template = existing_template

    ensure_chat_template(tokenizer)

    assert tokenizer.chat_template == existing_template


def test_default_chat_template_is_valid():
    """Test that DEFAULT_CHAT_TEMPLATE is a valid Jinja2 template."""
    # Basic sanity check: should contain Jinja2 template syntax and Llama 3 tokens
    assert "{%- for message in messages %}" in DEFAULT_CHAT_TEMPLATE
    assert "{%- if add_generation_prompt %}" in DEFAULT_CHAT_TEMPLATE
    assert "<|start_header_id|>" in DEFAULT_CHAT_TEMPLATE
    assert "<|eot_id|>" in DEFAULT_CHAT_TEMPLATE
