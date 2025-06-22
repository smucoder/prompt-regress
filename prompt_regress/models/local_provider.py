from ollama import Client, ChatResponse, AsyncClient
from .base import ModelProvider, ModelResponse

class LocalProvider(ModelProvider):
    def __init__(self, host: str):
        """
        Initialize the LocalProvider with the Ollama client.

        Args:
            url (str): The URL of the Ollama server. Defaults to "http://localhost:11434".
        """
        super().__init__()
        self.client = Client(
            host=host
        )
        self.async_client = AsyncClient()

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Get a completion for the given prompt using Ollama's API.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            ModelResponse: The model's response containing text and metadata.
        """
        response: ChatResponse = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=kwargs.pop('model'),
            options=kwargs
        )

        return ModelResponse(
            text=response.message.content,
            prompt=prompt,
            token_count=0,
            cost=0.0,
            response_time_ms=0.0,
            metadata={},
            raw_response=response
        )
    
    async def agenerate(self, prompt, **kwargs):
        """
        Asynchronously get a completion for the given prompt using Ollama's API.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            ModelResponse: The model's response containing text and metadata.
        """

        response: ChatResponse = await self.async_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=kwargs.pop('model'),
            options=kwargs
        )

        return ModelResponse(
            text=response.message.content,
            prompt=prompt,
            token_count=0,
            cost=0.0,
            response_time_ms=0.0,
            metadata={},
            raw_response=response
        )
    
    def input_tokens(self, prompt: str) -> int:
        """
        Calculate the number of input tokens for the given prompt.

        Args:
            prompt (str): The input prompt to analyze.

        Returns:
            int: The number of input tokens in the prompt.
        """
        pass
    
    def get_cost(self, input_tokens, output_tokens) -> float:
        """
        Calculate the cost for the given input and output tokens. 

        Args:
            input_tokens (int): The number of input tokens.
            output_tokens (int): The number of output tokens.
        
        Returns:
            float: The cost for the completion.
        """
        pass