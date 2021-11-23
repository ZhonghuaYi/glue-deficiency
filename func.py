import numpy as np


def log_transform(img):
    import math
    a = 255 / math.log(256)
    out = np.log(1 + img + 1e-5) * a
    return out.astype(np.uint8)
