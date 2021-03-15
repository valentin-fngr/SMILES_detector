import numpy as np 


def stepwise_scheduler(epochs, lr): 
    # modified regarding the learning curves
    if epochs < 50: 
        return lr
    else: 
        return lr * np.exp(-0.1)

