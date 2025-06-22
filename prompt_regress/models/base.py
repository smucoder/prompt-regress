from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelResponse:
    """Response from a model provider."""
    text: str
    prompt: str
    token_count: int
    cost: float
    response_time_ms: int
    metadata: Dict[str, Any]
    raw_response: Optional[Any] = None


class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Get a completion for the given prompt.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The model's completion for the prompt.
        """
        pass

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously get a completion for the given prompt.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The model's completion for the prompt.
        """
        pass


    @abstractmethod
    def input_tokens(self, prompt: str) -> int:
        """
        Calculate the number of input tokens for the given prompt.

        Args:
            prompt (str): The input prompt to analyze.

        Returns:
            int: The number of input tokens in the prompt.
        """
        pass


    @abstractmethod
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for the given input and output tokens.

        Args:
            input_tokens (int): The number of input tokens.
            output_tokens (int): The number of output tokens.

        Returns:
            float: The cost for the completion.
        """
        pass