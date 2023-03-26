# --------------------- Importing Libraries -------------------------

import json
import numpy as np
import random


# --------------------------- keeping all utility functions here --------------

# seeds
def set_seeds(seed=42):
    """Set seed for reproducibility"""

    np.random.seed(seed)
    random.seed(seed)


# load dictionary
def load_dict(filepath):
    """Load a dictionary from jsons filepath"""

    with open(filepath, "r") as fp:
        d = json.load(fp)

    return d


# save dictionary
def save_dict(d, filepath, cls=None, sortkeys=False):
    """Save a dictionary to a specific location."""

    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
