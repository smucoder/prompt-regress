import anthropic
import tiktoken
from .base import ModelProvider, ModelResponse

class AnthropicProvider(ModelProvider):
    def __init__(self):
        super().__init__()

        self.client = anthropic.Anthropic()
        self.encoding = tiktoken.encoding_for_model("gpt-4o")


    def generate(self, prompt: str, **kwargs) -> str:
        """
        Get a completion for the given prompt using Anthropic's API.

        Args:
            prompt (str): The input prompt to complete.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The model's completion for the prompt.
        """
        message = self.client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            **kwargs
            )


        return ModelResponse(
            text=message['content'][0]['text'],
            prompt=prompt,
            token_count=0,
            cost=0,
            response_time_m=0,
            metadata={},
            raw_response=message
            )
        


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