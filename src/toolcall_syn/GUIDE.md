# LLM Client Usage Guide

This guide explains how to use the `llm_client.py` module, which provides functionality for interacting with various LLM APIs.

## Overview

The `llm_client.py` module contains a class called `LLMJudge` that allows you to send prompts to language models and receive responses. It supports multiple API providers including OpenAI, Azure OpenAI, and Microsoft Research Azure deployments.

## Configuration

### LLMJudgeConfig

The `LLMJudgeConfig` class is used to configure the LLM client:

```python
from llm_client import LLMJudgeConfig, LLMJudge

# Create a configuration
config = LLMJudgeConfig(
    api_key="your_api_key",  # Default: "123"
    base_url="http://your-api-endpoint:port/v1",  # Default: "http://0.0.0.0:8112/v1"
    temperature=0.3,  # Default: 0.3
    max_tokens=1024,  # Default: 1024
    model_name=None  # Default: None (will be auto-detected)
)
```

## Environment Variables

The client supports various environment variables to configure API access:

- `OPENAI_API_TYPE`: Set to "openai", "azure_key", or "azure_msra" to specify the API provider
- `OPENAI_API_KEY`: Your API key
- `OPENAI_API_BASE`: Base URL for the API
- `OPENAI_API_MODEL`: Model name to use
- `OPENAI_API_VERSION`: API version (for Azure)
- `OPENAI_DEPLOYMENT_NAME`: Deployment name (for Azure)
- `OPENAI_INSTANCE`: Instance name (for Azure MSRA)

## Basic Usage

### Initialization

```python
from llm_client import LLMJudge, LLMJudgeConfig

# Using default configuration
judge = LLMJudge()

# Or with custom configuration
config = LLMJudgeConfig(temperature=0.7, max_tokens=2048)
judge = LLMJudge(config)
```

### Single Prediction

To send a single prompt to the LLM:

```python
prompt = "Explain quantum computing in simple terms."
response = judge.predict(prompt)
print(response)
```

The `predict` method includes automatic retry logic (3 retries with 10-second delays) in case of API failures.

### Batch Prediction

To process multiple prompts efficiently:

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "How does natural language processing work?"
]
responses = judge.batch_predict(prompts)

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
```

The `batch_predict` method processes prompts in batches of 5 (configurable) and includes the same retry mechanism as the single prediction method.

## Cost Tracking

The `LLMJudge` class automatically tracks token usage and associated costs:

```python
# After making predictions, you can save cost information
cost_data = judge.save_cost()
print(f"Input tokens: {cost_data['total_input_tokens']}")
print(f"Output tokens: {cost_data['total_output_tokens']}")
print(f"Total cost: ${cost_data['total_cost']}")
```

Cost information is automatically saved to a `cost.jsonl` file when using `batch_predict`.

## Utility Functions

### Extract XML Content

The class provides a utility method to extract content from XML tags:

```python
text = "<answer>This is the answer</answer>"
success, content = LLMJudge._extract_xml_content(text, "answer")
if success:
    print(content)  # Outputs: "This is the answer"
```

## Supported Models

The client supports various models with different pricing:

- GPT-4o
- GPT-4o-mini
- O4-mini
- O3-mini
- O1
- O1-mini
- GPT-3.5-turbo
- GPT-35-turbo

## Error Handling

The client includes robust error handling with automatic retries for API failures. If all retries fail:
- `predict()` returns an empty string
- `batch_predict()` returns empty strings for failed items in the batch
