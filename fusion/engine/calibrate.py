"""engine/calibrate.py: turns the winner's vote share into a calibrated probability."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

ESCAPE = 0.5                                     # below this probability, the question goes to Opus


def fit(share, right):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(share, right)


def ece(p, y, bins=10):
    """Expected calibration error: the gap between confidence and accuracy per bin, weighted by the bin's size."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in set(b)))
