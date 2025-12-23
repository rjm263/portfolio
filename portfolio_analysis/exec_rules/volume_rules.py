import pandas as pd
import numpy as np

def nv(vol: pd.Series):
    m = np.sign(vol.diff())
    return np.array(m - 1, dtype=bool)
    

VOLUME_RULES = {None: [], 'neg_vol': nv}