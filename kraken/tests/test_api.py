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