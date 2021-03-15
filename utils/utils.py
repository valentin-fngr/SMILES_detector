import numpy as np 

def stepwise_scheduler(epochs, lr): 
    # modified regarding the learning curves
    if epochs < 12: 
        return lr
    if epochs >= 12 and epochs <= 15: 
        return lr * np.exp(-0.13)
    else: 
        return lr * np.exp(-0.22)

