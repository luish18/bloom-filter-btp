import json
import os

import cv2 as cv
import numpy as np
import numpy.typing as npt

from bloom_filter_btp.img_processing import (
    apply_gabor_filter,
    capture_and_extract_face,
    divide_into_blocks,
    extract_face,
)

N_GROUPS = 10
N_BLOCKS = 80
MIN_NEIGHBORS = 5

SCALE_FACTOR = 1.1
IMG_SHAPE = (300, 300)

TEMPLATE_PATH = "data/templates/user_templates.json"


def compare_bloom_filters(bf1: npt.NDArray, bf2: npt.NDArray) -> int:
    """
    Compare two bloom filters using Hamming distance.

    Parameters:
    bf1 (list or array-like): The first bloom filter.
    bf2 (list or array-like): The second bloom filter.

    Returns:
    int: The Hamming distance between the two bloom filters.
    """
    if len(bf1) != len(bf2):
        raise ValueError("Bloom filters must be of the same length")

    return np.count_nonzero(bf1 != bf2)


def structure_preserving_rearrangement(
    blocks: list[npt.NDArray], nGroups: int
) -> list[npt.NDArray[np.uint8]]:
    """
    Perform structure-preserving feature re-arrangement on biometric template blocks.

    Parameters:
    - blocks (List[numpy.ndarray]): List of 2D arrays representing the blocks of the biometric template.
    - nGroups (int): The number of groups to divide the blocks into.

    Returns:
    - rearranged_blocks (List[numpy.ndarray]): List of rearranged blocks after the structure-preserving process.

    This function rearranges the blocks by grouping, concatenating, and permuting rows within each group.
    """

    nBlocks = len(blocks)
    nBlocksGroup = nBlocks // nGroups

    if nBlocks % nGroups != 0:
        raise ValueError(
            "The total number of blocks must be evenly divisible by nGroups."
        )

    # Random number generator for permutation
    rng = np.random.default_rng()

    # Split blocks into groups
    groups = [blocks[i * nBlocksGroup : (i + 1) * nBlocksGroup] for i in range(nGroups)]

    rearranged_blocks = []

    for group in groups:
        # Vertically concatenate blocks in the group
        concatenated_group = np.vstack(group)

        # Permute the rows of the concatenated group
        rng.shuffle(concatenated_group)

        # Split and add the permuted rows back to rearranged_blocks
        rearranged_blocks.extend(np.array_split(concatenated_group, nBlocksGroup))

    return rearranged_blocks


def compute_bloom_filters(blocks: list[npt.NDArray]) -> npt.NDArray:
    """
    Compute Bloom filters from the given blocks, with each filter having a size of 2^nBits.

    Parameters:
    - blocks (List[numpy.ndarray]): List of 2D arrays representing the blocks.
    - nBits (int): The number of bits used to determine the size of the Bloom filter, with actual size being 2^nBits.

    Returns:
    - bloom_filters (numpy.ndarray): An array containing the Bloom filters.
    """

    nBlocks = len(blocks)
    nBits = blocks[0].shape[0]
    bloom_filter_size = 2**nBits
    bloom_filters = np.zeros((nBlocks, bloom_filter_size), dtype=int)

    # binarize blocks
    for block in blocks:
        block[block > 0] = 1

    for i, block in enumerate(blocks):
        for col in range(block.shape[1]):
            # Translate column to decimal value
            decimal_value = int("".join(map(str, block[:, col])), 2)

            # Set the corresponding bit in the Bloom filter
            bloom_filters[i, decimal_value % bloom_filter_size] = 1

    return bloom_filters


def gen_template(face: npt.NDArray[np.uint8] | cv.Mat) -> npt.NDArray[np.uint8]:
    features = apply_gabor_filter(face)
    blocks = divide_into_blocks(features, N_BLOCKS)
    blocks = structure_preserving_rearrangement(blocks, N_GROUPS)
    template = compute_bloom_filters(blocks)

    return template


def save_user_template(
    username: str, template: npt.NDArray, template_path: str = TEMPLATE_PATH
) -> None:
    """
    Save the user template to a file.

    This function takes a username, a template, and an optional template path as input.
    It saves the user template to the specified template path in JSON format.

    Parameters
    ----------
    username : str
        The username of the user.
    template : npt.NDArray
        The template to be saved.
    template_path : str, optional
        The path where the template will be saved, by default TEMPLATE_PATH.
    """
    try:
        with open(template_path, "r") as file:
            templates = json.load(file)
    except FileNotFoundError:
        templates = {}

    templates[username] = template.tolist()

    with open(template_path, "w") as file:
        json.dump(templates, file)


def load_user_templates() -> dict:
    """
    Load user templates from a JSON file.

    Returns:
        dict: A dictionary containing the loaded user templates.
    """
    try:
        with open("user_templates.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def register_user() -> None:
    """
    Registers a user by capturing their face, extracting face features,
    and saving the user template.

    Returns:
        None
    """
    username = input("Enter your name: ")
    print("Please, look at the camera...")

    face = None
    while face is None:
        face, _ = capture_and_extract_face(
            scale_factor=SCALE_FACTOR, final_size=IMG_SHAPE, min_neighbors=MIN_NEIGHBORS
        )
        if face is None:
            print("No face detected. Please try again.")

    print("Extracting face features...")
    template = gen_template(face)

    print("Saving user template...")
    save_user_template(username, template)

    print("User registered successfully!")


def check_user_in_db() -> None:
    """
    Checks if a user is in the database by comparing their face template with stored templates.

    This function prompts the user to look at the camera and captures their face. It then extracts
    the face features and compares them with the stored templates of registered users. The function
    prints the username and the distance between the new template and each stored template.

    If no user templates are found, it prints a message asking the user to register first. If no
    face is detected during the capture process, it prints a message asking the user to try again.

    Returns:
        None
    """
    print("Please, look at the camera...")

    if not os.path.exists(TEMPLATE_PATH):
        print("No user templates found. Please register first.")
        return

    face = None
    while face is None:
        face, _ = capture_and_extract_face(
            scale_factor=SCALE_FACTOR, final_size=IMG_SHAPE, min_neighbors=MIN_NEIGHBORS
        )

        if face is None:
            print("No face detected. Please try again.")

    print("Extracting face features...")
    new_template = gen_template(face)

    print("Loading all user templates...")
    templates = load_user_templates()

    print("Checking if user is in the database...")
    for username, template in templates.items():
        # Compare the new template with each stored template.
        # Replace this with your actual implementation.
        distance = compare_bloom_filters(new_template, template)
        print(f"User found: {username} with distance: {distance}")


def find_img_file(path: str) -> None:
    """
    Finds the face in an image file and compares it with stored templates in the database.

    Args:
        path (str): The path to the image file.

    Returns:
        None
    """
    print("Extracting face features...")

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    face = extract_face(
        img,
        final_size=IMG_SHAPE,
        scale_factor=SCALE_FACTOR,
        min_neighbors=MIN_NEIGHBORS,
    )

    if face is None:
        print("No face detected in image")
        return

    new_template = gen_template(face)

    print("Loading all user templates...")
    templates = load_user_templates()

    print("Checking if user is in the database...")
    for username, template in templates.items():
        # Compare the new template with each stored template.
        # Replace this with your actual implementation.
        distance = compare_bloom_filters(new_template, template)
        print(f"User found: {username} with distance: {distance}")


def register_image_file(path: str) -> None:
    """
    Register an image file by extracting face features, generating a template, and saving it.

    Args:
        path (str): The path to the image file.

    Returns:
        None
    """
    print("Extracting face features...")

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    face = extract_face(
        img,
        final_size=IMG_SHAPE,
        scale_factor=SCALE_FACTOR,
        min_neighbors=MIN_NEIGHBORS,
    )

    if face is None:
        print("No face detected in image")
        return

    new_template = gen_template(face)

    print("Saving user template...")
    save_user_template(path, new_template)

    print("User registered successfully!")
