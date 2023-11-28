from typing import Optional

import cv2 as cv
import numpy as np
import numpy.typing as npt
from cv2.data import haarcascades as haarcascades


def divide_into_blocks(
    matrix: npt.NDArray[np.uint8] | cv.Mat, N_words: int, N_bits: int
) -> list[npt.NDArray[np.uint8]]:
    """
    Divide a matrix into blocks with specified dimensions.

    Parameters:
    - matrix (numpy.ndarray): The 2D input matrix (feature matrix).
    - N_words (int): The number of columns in each block.
    - N_bits (int): The number of rows in each block.

    Returns:
    - blocks (list of numpy.ndarray): A list containing the blocks.
    """
    flattened = matrix.flatten()

    total_blocks = len(flattened) // (N_words * N_bits)

    if len(flattened) % (N_words * N_bits) != 0:
        raise ValueError(
            "Matrix cannot be evenly divided into the specified block size."
        )

    blocks = [
        flattened[i * N_words * N_bits : (i + 1) * N_words * N_bits].reshape(
            N_bits, N_words
        )
        for i in range(total_blocks)
    ]

    return blocks


def apply_gabor_filter(
    image: npt.NDArray[np.uint8] | cv.Mat,
    ksize: tuple[int, int] = (50, 50),
    sigma: int = 8,
    theta: float = np.pi / 4,
    lambd: int = 10,
    gamma: float = 0.5,
    psi: int = 0,
) -> npt.NDArray[np.uint8]:
    """
    Apply a Gabor filter to an image.

    Parameters:
    - image (numpy.ndarray): The input image in grayscale format.
    - ksize (tuple of int): Size of the Gabor filter kernel. Default is (50, 50).
    - sigma (float): Standard deviation of the Gaussian envelope. Default is 8.
    - theta (float): Orientation of the normal to the parallel stripes of the Gabor function. Default is π/4.
    - lambd (float): Wavelength of the sinusoidal factor. Default is 10.
    - gamma (float): Spatial aspect ratio. Default is 0.5.
    - psi (float): Phase offset. Default is 0.

    Returns:
    - filtered_image (numpy.ndarray): The image after applying the Gabor filter.

    This function creates a Gabor filter with the specified parameters and applies it to the provided image.
    """
    gabor_kernel = cv.getGaborKernel(
        ksize, sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F
    )

    filtered_image = cv.filter2D(image, -1, gabor_kernel)
    filtered_image = filtered_image.astype(np.uint8)

    return filtered_image


def show_img(img: cv.Mat) -> None:
    cv.imshow("Capture", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extract_face(
    img: cv.Mat, scale_factor: float = 1.5, min_neighbors: int = 5
) -> Optional[npt.NDArray[np.uint8]]:
    """
    Extracts the face from the image using opencv's haarcascade classifier.

    Args:
        img (cv.Mat): The image from which to extract the face.

    Returns:
        cv.Mat: The extracted face.
    """

    face_cascade = cv.CascadeClassifier(
        haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        img, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    if len(faces) == 0:
        print("No face detected")
        return None
    elif len(faces) > 1:
        print("More than one face detected")
        return None

    x, y, w, h = faces[0]

    return img[y : y + h, x : x + w]


def capture_image(exit_key: str = "q", capture_key: str = "c") -> cv.Mat:
    cap = cv.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    done = False
    while not done:
        ret, frame = cap.read()
        frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imshow("Input", frame)

        c = cv.waitKey(1)
        if c == ord(exit_key):
            result = None
            done = True

        elif c == ord(capture_key):
            result = frame
            done = True

    cap.release()
    cv.destroyAllWindows()

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    return result
