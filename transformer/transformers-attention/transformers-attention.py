import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    
    assert Q.size(-1) == K.size(-1) == V.size(-1)
    d_k = Q.size(-1)
    
    S = Q @ K.transpose(-2, -1) 
    
    scale_factor = 1 / (math.sqrt(d_k))

    S_scaled = S * scale_factor
    
    soft_max = F.softmax(S_scaled, dim=-1)
    
    weight = soft_max @ V
    return weight