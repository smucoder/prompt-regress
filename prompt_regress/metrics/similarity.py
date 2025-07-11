import torch
import numpy as np

from typing import List
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process


class SimilarityMetrics:
    """
    A class to calculate text similarity metrics.
    
    This class provides methods to compute both character-based and semantic similarity
    between two texts using a pre-trained model.
    """
    
    def __init__(self, embedding_model: str, batch_size: int = 16):
        """
        Initialize the SimilarityMetrics class with a pre-trained embedding model.

        Args:
            embedding_model (str): The name of the pre-trained model to use for semantic similarity.
                                   Default is "Qwen/Qwen3-Embedding-0.6B".
        """
        self.model = SentenceTransformer(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size


    def text_similarity(self, baseline_texts: List[str], target_texts: List[str]) -> List[float]:
        """
        Calculate the similarity between two texts using a simple character-based ratio.
        
        Args:
            text1 (str): The first text.
            text2 (str): The second text.
        
        Returns:
            float: A similarity score between 0 and 1.
        """
        if not baseline_texts or not target_texts:
            raise ValueError("Both baseline_texts and target_texts must be non-empty lists.")
        similarity_matrix = process.cdist(baseline_texts, target_texts, scorer=fuzz.ratio)
        one_to_one = np.diag(similarity_matrix) / 100.0
        return one_to_one


    def semantic_similarity(self, baseline_texts: List[str], target_texts: List[str]) -> List[float]:
        """
        Calculate the semantic similarity between two texts using a pre-trained model.
        
        Args:
            text1 (str): The first text.
            text2 (str): The second text.
        
        Returns:
            float: A similarity score between 0 and 1.
        """
        if not baseline_texts or not target_texts:
            raise ValueError("Both baseline_texts and target_texts must be non-empty lists.")
        baseline_embeddings = self.model.encode(baseline_texts, batch_size=self.batch_size, convert_to_tensor=True)
        target_embeddings = self.model.encode(target_texts, batch_size=self.batch_size, convert_to_tensor=True)
        similarities = torch.cosine_similarity(baseline_embeddings, target_embeddings, dim=1)
        similarities = similarities.cpu().numpy()
        return similarities


if __name__ == "__main__":
    texts1 = []
    texts2 = ["b"]

    metrics = SimilarityMetrics(embedding_model='Qwen/Qwen3-Embedding-0.6B')
    
    print("Text Similarity:", metrics.text_similarity(texts1, texts2))
    print("Semantic Similarity:", metrics.semantic_similarity(texts1, texts2))