import tiktoken
from openai import OpenAI, AsyncOpenAI
from .base import ModelProvider, ModelResponse

class OpenAIProvider(ModelProvider):
    def __init__(self):
        super().__init__()
        self.client = OpenAI()
        self.async_client =  AsyncOpenAI()

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")


    def generate(self, prompt, **kwargs) -> ModelResponse:
        """
        Get a completion for the given prompt using OpenAI's API.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The model's completion for the prompt.
        """
        response = self.client.responses.create(
            input=prompt,
            **kwargs
        )

        return ModelResponse(
            text=response.output_text,
            prompt=prompt,
            token_count=0,
            cost=0,
            response_time_m=0,
            metadata={},
            raw_response=response
        )
    
    async def agenerate(self, prompt, **kwargs) -> ModelResponse:
        """
        Asynchronously get a completion for the given prompt using OpenAI's API.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The model's completion for the prompt.
        """
        response = await self.async_client.responses.create(
            input=prompt,
            **kwargs
        )

        return ModelResponse(
            text=response.output_text,
            prompt=prompt,
            token_count=0,
            cost=0,
            response_time_m=0,
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
        return len(self.encoding.encode(prompt, allowed_special="all"))
    


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