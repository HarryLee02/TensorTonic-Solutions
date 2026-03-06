import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    PE = []
    for idx in range(seq_length):
        for _dim in range(0, d_model, 2):
            PE.append([np.sin(idx/(10000**(idx/d_model))),np.cos(idx/(10000**(idx/d_model)))])
        

        
    PE = np.reshape(PE,(seq_length,d_model))

    return PE
    