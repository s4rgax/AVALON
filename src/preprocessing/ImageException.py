class ImageException(Exception):
    
    """
    ImageException allows exceptions to be raised in certain error situations
    that may occur while performing operations on images.
    """
    def __init__(self, msg: str) -> None:
        super().__init__(msg)