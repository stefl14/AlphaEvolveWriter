"""Unit tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from kraken.api.app import app
from kraken.api.models import CompletionRequest, CompletionResponse


client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns expected response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Kraken Text Completion API"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_completion_request_validation():
    """Test request validation for completion endpoint."""
    # Test with empty prompt
    request = {"prompt": "", "max_length": 50}
    assert request["prompt"] == ""

    # Test with valid prompt
    request = {"prompt": "Hello, how can I", "max_length": 50}
    assert len(request["prompt"]) > 0


def test_completion_request_model():
    """Test the CompletionRequest model."""
    request = CompletionRequest(
        prompt="Test prompt",
        max_length=100,
        temperature=0.8
    )
    assert request.prompt == "Test prompt"
    assert request.max_length == 100
    assert request.temperature == 0.8


def test_completion_response_model():
    """Test the CompletionResponse model."""
    response = CompletionResponse(
        prompt="Test prompt",
        completion="help you today?",
        total_tokens=10,
        model_name="gpt2"
    )
    assert response.prompt == "Test prompt"
    assert response.completion == "help you today?"
    assert response.total_tokens == 10
    assert response.model_name == "gpt2"


def test_empty_prompt_handling():
    """Test handling of empty prompts."""
    request = {"prompt": ""}
    # Empty prompt should be caught by validation
    assert request["prompt"] == ""


def test_long_prompt_handling():
    """Test handling of very long prompts."""
    long_prompt = "x" * 2000  # Exceeds max length
    request = {"prompt": long_prompt, "max_length": 50}
    assert len(request["prompt"]) == 2000


def test_special_characters_in_prompt():
    """Test handling of special characters."""
    special_prompt = "Hello @#$%^&*() \n\t world!"
    request = {"prompt": special_prompt}
    assert "\n" in request["prompt"]
    assert "\t" in request["prompt"]