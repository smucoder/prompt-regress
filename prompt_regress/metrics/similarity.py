import torch

from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher


class SimilarityMetrics:
    """
    A class to calculate text similarity metrics.
    
    This class provides methods to compute both character-based and semantic similarity
    between two texts using a pre-trained model.
    """
    
    def __init__(self, embedding_model: str):
        """
        Initialize the SimilarityMetrics class with a pre-trained embedding model.

        Args:
            embedding_model (str): The name of the pre-trained model to use for semantic similarity.
                                   Default is "Qwen/Qwen3-Embedding-0.6B".
        """
        self.model = SentenceTransformer(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")


    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts using a simple character-based ratio.
        
        Args:
            text1 (str): The first text.
            text2 (str): The second text.
        
        Returns:
            float: A similarity score between 0 and 1.
        """
        ratio = SequenceMatcher(None, text1, text2).ratio()
        return round(ratio, 2)


    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the semantic similarity between two texts using a pre-trained model.
        
        Args:
            text1 (str): The first text.
            text2 (str): The second text.
        
        Returns:
            float: A similarity score between 0 and 1.
        """
        embeddings = self.model.encode([text1, text2])
        similarity = self.model.similarity(embeddings[0], embeddings[1])

        return round(similarity.item(), 2)


if __name__ == "__main__":
    text1 = "This is a test sentence."
    text2 = "This is a another test sentence."

    metrics = SimilarityMetrics(embedding_model='Qwen/Qwen3-Embedding-0.6B')
    
    print("Text Similarity:", metrics.text_similarity(text1, text2))
    print("Semantic Similarity:", metrics.semantic_similarity(text1, text2))