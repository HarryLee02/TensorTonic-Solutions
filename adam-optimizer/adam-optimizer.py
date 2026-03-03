import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m1 = np.dot(beta1, m) + np.dot((1 - beta1), grad)
    v1 = np.dot(beta2, v) + np.dot((1 - beta2), np.square(grad))

    m_correct = m1 / (1 - beta1**t)
    v_correct = v1 / (1 - beta2**t)

    param_new = param - np.dot(lr,( m_correct / (np.sqrt(v_correct) + eps)))

    return (param_new, m1, v1)