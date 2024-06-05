"""
File contenente l'implementazione di un early stopper, da usare durante l'addestramento
di un modello CNN, attraverso la definizione dell'omonima classe EarlyStopper.
"""

import numpy as np

class EarlyStopper:
    """
    Metodo costruttore della classe.
    """
    def __init__(self, patience: int = 1, minDelta: int = 0) -> None:
        self.patience = patience
        self.minDelta = minDelta
        self.counter = 0
        self.minValidationLoss = np.inf
    
    """
    Metodo che in base al valore di validationLoss passato in input restituisce 
    TRUE (l'addestramento deve essere interrotto) o FALSE (l'addestramento può continuare).
    """

    def earlyStop(self, validationLoss: float) -> bool:
        if validationLoss < self.minValidationLoss - self.minDelta:
            self.minValidationLoss = validationLoss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False