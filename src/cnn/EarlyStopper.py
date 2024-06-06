import numpy as np

class EarlyStopper:
    """
    Constructor method for early stopping
    :param patience: num of epochs minimum to enable early stopping
    :param minDelta: minimum delta for early stop
    """
    def __init__(self, patience: int = 1, minDelta: int = 0) -> None:
        self.patience = patience
        self.minDelta = minDelta
        self.counter = 0
        self.minValidationLoss = np.inf
    
    def earlyStop(self, validationLoss: float) -> bool:
        if validationLoss < self.minValidationLoss - self.minDelta:
            self.minValidationLoss = validationLoss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False