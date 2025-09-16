"""Data preparation utilities for training the text completion model."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare text data for model training."""

    def __init__(self, data_path: str):
        """Initialize DataLoader with path to data file.

        Args:
            data_path: Path to the JSON data file
        """
        self.data_path = Path(data_path)
        self.data = []

    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file.

        Returns:
            List of data records
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return []

        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} records from {self.data_path}")

        return self.data

    def prepare_prompts(self) -> List[str]:
        """Extract and prepare prompts from the data.

        Returns:
            List of text prompts for training
        """
        prompts = []
        for record in self.data:
            if 'text' in record:
                prompts.append(record['text'])

        logger.info(f"Prepared {len(prompts)} prompts for training")
        return prompts