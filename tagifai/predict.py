# ------------------------------ Importing Libraries --------------------------

import numpy as np


# ---------------------- custom predict function --------------------------

def custom_predict(y_prob, threshold, index):
    """Custom predict function that defaults to an index if conditions are not met"""

    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)
