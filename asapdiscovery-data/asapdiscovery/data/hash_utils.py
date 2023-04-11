import hashlib
import pathlib


def calculate_sha256_file(file_path: pathlib.Path) -> str:
    """
    Calculates the SHA-256 hash of a file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the file to calculate the hash of.

    Returns
    -------
    str
        SHA-256 hash of the file.
    """
    with open(file_path, "rb") as f:
        hash_object = hashlib.sha256()
        while True:
            data = f.read(8192)
            if not data:
                break
            hash_object.update(data)
    return hash_object.hexdigest()


def compare_file_sha256_hashes(file_path: pathlib.Path, expected_hash: str) -> None:
    """
    Compares the SHA-256 hash of a file to an expected hash.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the file to calculate the hash of.
    expected_hash : str
        Expected hash of the file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the calculated hash does not match the expected hash.
    """
    if not calculate_sha256_file(file_path) == expected_hash:
        raise ValueError(
            f"Hash of {file_path} does not match expected hash {expected_hash}."
        )
