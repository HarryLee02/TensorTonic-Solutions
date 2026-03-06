import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    emb_layer = nn.Embedding(vocab_size, d_model)
    return emb_layer

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Your code here
    e = torch.Tensor(embedding(tokens)) * (d_model**(1/2)) #scale
    
    return e