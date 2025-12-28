"""Image preprocessing utilities."""

import cv2
import numpy as np
from typing import Tuple


def resize_image(image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """Resize image if it exceeds maximum dimensions while preserving aspect ratio.

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    # Check if resizing is needed
    if width <= max_width and height <= max_height:
        return image

    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized


def preprocess_image(image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    """Apply preprocessing to reduce noise.

    Args:
        image: Input image
        blur_kernel: Gaussian blur kernel size (must be odd)

    Returns:
        Preprocessed image
    """
    # Apply Gaussian blur to reduce noise
    if blur_kernel > 0 and blur_kernel % 2 == 1:
        blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    else:
        blurred = image

    return blurred


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV color space.

    Args:
        image: Input BGR image

    Returns:
        HSV image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale.

    Args:
        image: Input BGR image

    Returns:
        Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to handle varying lighting.

    Args:
        image: Input grayscale image
        block_size: Size of pixel neighborhood
        c: Constant subtracted from mean

    Returns:
        Binary threshold image
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE.

    Args:
        image: Input grayscale or BGR image

    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space for color images
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Apply CLAHE directly to grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def create_mask_from_hsv(
    hsv_image: np.ndarray,
    lower_bound: Tuple[int, int, int],
    upper_bound: Tuple[int, int, int]
) -> np.ndarray:
    """Create binary mask from HSV color range.

    Args:
        hsv_image: Input image in HSV color space
        lower_bound: Lower HSV bound (H, S, V)
        upper_bound: Upper HSV bound (H, S, V)

    Returns:
        Binary mask where pixels in range are white
    """
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    return cv2.inRange(hsv_image, lower, upper)


def morphological_close(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological closing to fill small holes.

    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel

    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def morphological_open(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological opening to remove small noise.

    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel

    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
