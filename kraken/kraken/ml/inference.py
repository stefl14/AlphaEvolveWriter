"""Model inference utilities for text completion."""

import logging
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kraken.ml.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelInference:
    """Handle model loading and text generation."""

    def __init__(self, config: ModelConfig):
        """Initialize the inference engine.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.config.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_completion(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate text completion for a single prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Dictionary with completion and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from completion
        if completion.startswith(prompt):
            completion = completion[len(prompt):].strip()

        return {
            "prompt": prompt,
            "completion": completion,
            "total_tokens": len(outputs[0]),
            "model_name": self.config.model_name
        }