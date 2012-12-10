import numpy as np
from rely import reliability, prepare_reliability
from performance import performance_diagram


def combine_bins(fcst, obs, num):
    """
    Aggregates data from higher resolution to lower resolution

    Parameters
    ----------
    fcst : array_like
        The higher resolution forecast probabilities
    obs : array_like
        The higher resolution observed probabilities
    num : number
        The number of bins used after combining probabilities

    Returns
    -------
    new_fcst : array_like
        The aggregated forecast probabilities
    new_obs : array_like
        The aggregated observed probabilities

    """
    fcst = np.asarray(fcst)
    obs = np.asarray(obs)
    orig_size = fcst.shape[0] - 1
    new_size = int(orig_size / num) + 2
    new_fcst = np.zeros(new_size, dtype=float)
    new_obs = np.zeros(new_size, dtype=float)
    for i in range(0, new_size-3):
        a = i*num + 1
        b = a + num
        new_fcst[i+1] = fcst[a:b].sum()
        new_obs[i+1] = obs[a:b].sum()
    i += 1
    a = i*num + 1
    b = a + num
    new_fcst[i+1] = fcst[a:b].sum()
    new_obs[i+1] = obs[a:b].sum()
    new_fcst[0] = fcst[0]
    new_obs[0] = obs[0]
    new_fcst[-1] = fcst[-1]
    new_obs[-1] = obs[-1]
    return new_fcst, new_obs
