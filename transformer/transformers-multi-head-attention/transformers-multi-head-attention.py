import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    batch_size = Q.shape[0] 

    assert Q.shape[1] == K.shape[1] == V.shape[1]
    seq_len = Q.shape[1] 
    
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
    d_model = Q.shape[-1] 
    
    assert d_model % num_heads == 0
    d_head = d_model // num_heads

    Q_full = Q @ W_q
    K_full = K @ W_k
    V_full = V @ W_v

    Q_full = Q_full.reshape(batch_size,seq_len,num_heads,d_head).transpose(0,2,1,3)
    K_full = K_full.reshape(batch_size,seq_len,num_heads,d_head).transpose(0,2,1,3)
    V_full = V_full.reshape(batch_size,seq_len,num_heads,d_head).transpose(0,2,1,3)

    score = Q_full @ np.transpose(K_full,(0,1,3,2)) * (1/np.sqrt(d_head))

    score = softmax(score) @ V_full

    score = np.transpose(score,(0,2,1,3))

    score = score.reshape(batch_size, seq_len, d_model)
    
    result = score @ W_o
    
    return result
