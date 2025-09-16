# Kraken Text Completion API

An intelligent text auto-completion API for customer service messages, built for Octopus Energy's Kraken system.

## Features

- Single and batch text completion
- Configurable generation parameters
- Health monitoring endpoints
- OpenAPI documentation

## Quick Start

```bash
# Install dependencies
pip install -e .

# Start the API server
uvicorn kraken.api.app:app --reload

# Test the API
curl -X POST "http://localhost:8000/v1/complete" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how can I"}'
```

## API Documentation

Once running, visit:
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

- `POST /v1/complete` - Generate single completion
- `POST /v1/complete/batch` - Generate multiple completions
- `GET /health` - Health check
- `GET /ready` - Readiness check