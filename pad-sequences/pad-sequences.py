import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    seq_len = [len(seq) for seq in seqs]

    L= max_len if max_len !=None else max(seq_len)

    array = []
    
    for i, length in enumerate(seq_len):
        
        if length >= L:
            array.append(seqs[i][:L])
        elif length < L:
            array.append(seqs[i] + ([pad_value] * (L - length)))
            
    result = np.array(array,np.int32)
    
    return result