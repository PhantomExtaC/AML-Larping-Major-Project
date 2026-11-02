import hashlib
import numpy as np

def make_account_id(bank: str, account: str) -> str:
    """
    Standardize account ID formatting.
    Verify separator matches scripts/data/preprocess_hi_small.py
    """
    return f"{bank}-{account}"

def derive_rng(seed: int, *labels: str) -> np.random.Generator:
    """
    Derives a deterministic numpy Random Generator from a base seed and string labels.
    """
    text = f"{seed}_" + "_".join(labels)
    # Hash to a 32-bit integer for numpy seed
    hash_int = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    return np.random.default_rng(hash_int)