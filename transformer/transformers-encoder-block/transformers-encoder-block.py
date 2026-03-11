import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """

    d_model = x.shape[-1]

    mean = np.mean(x, axis = -1, keepdims = True)

    variance = np.var(x, axis = -1, keepdims = True)

    x_ = gamma * (x - mean) / np.sqrt(variance + eps) + beta
    return x_

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    d_model = Q.shape[2]
    batch_size = Q.shape[0]
    seq_len = Q.shape[1]
    d_k = d_model // num_heads

    Q_full = Q @ W_q
    K_full = K @ W_k
    V_full = V @ W_v
    
    Q_full = Q.reshape(batch_size,seq_len, num_heads, d_k).transpose(0,2,1,3)
    K_full = K.reshape(batch_size,seq_len, num_heads, d_k).transpose(0,2,1,3)
    V_full = V.reshape(batch_size,seq_len, num_heads, d_k).transpose(0,2,1,3)


    score = Q_full @ np.transpose(K_full,(0,1,3,2)) / np.sqrt(d_k)
    score = softmax(score) @ V_full
    
    score = np.transpose(score, (0,2,1,3)).reshape(batch_size,seq_len, d_model)

    score = score @ W_o
    return score

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h = x @ W1 + b1
    h_ = np.maximum(0, h)

    h_ = h_ @ W2 + b2

    return h_

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    mul_attn = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    x_ = layer_norm(x + mul_attn, gamma1, beta1)

    output = layer_norm(x_ + feed_forward(x_, W1, b1, W2, b2), gamma2, beta2)

    return output
    