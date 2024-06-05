"""
File contenente la definizione della classe ImageException.
ImageException permette di sollevare delle eccezioni in determinate situazioni di errore
che si possono verificare durante lo svolgimento di operazioni su immagini.
"""

class ImageException(Exception):
    
    """
    Metodo costruttore.
    """
    def __init__(self, msg: str) -> None:
        super().__init__(msg)