"""
Useful tools for personal_trainer package.

Author: Hanna L.A.
Date: June 2022
"""
import numpy as np
import cv2


def n_resize(img: np.ndarray,
             n: int = 3) -> np.ndarray:
    """
    Resize an image dividing image shape by n.

    Args:
        img: A three channel RGB image represented as numpy ndarray.
        n: Number by which image shape will be divided.

    Returns:
        Resized image.

    """
    height, width, _ = img.shape
    resize = 1 / n
    new_size = int(width * resize), int(height * resize)
    return cv2.resize(img, new_size)
