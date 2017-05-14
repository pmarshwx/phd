"""
A collection of various meteorological verification scores.
"""
from __future__ import print_function, division
import numpy as np
from veripy import components


def brier(fcsts, obs, fcst_percent, obs_percent):
    """
    A function to compute the Brier Score.
    
    Parameters
    ----------
    fcsts : numpy array               
        Array of the number of forecasts at a given forecast percent
    obs : numpy array
        Array of the number of observed grid points at a given 
        forecast threshold
    fcst_percent : numpy array
        An ordered array of the forecast probabilities.
    obs_percent : numpy array
        The ratio of observation points to forecast points at the
        corresponding forecast percent (fcst_percent)

    Returns
    -------
    brier_score : float
        The Brier Score
    """
    
    reliability = components.get_reliability(fcsts, fcst_percent, obs_percent)
    resolution = components.get_resolution(fcsts, obs, obs_percent)
    uncertainty = components.get_uncertainty(fcsts, obs)
    
    return (reliability - resolution + uncertainty)


def test_brier():
    """ Test of the Brier Score """
    
    import warnings
    warnings.simplefilter('error', Warning)
    
    fcst_percent = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, \
                             0.9, 1.])

    fcounts = np.array([1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 0], dtype=float)
    obs = np.array([0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0])
    obs_percent = obs/fcounts
    obs_percent[np.isnan(obs_percent)] = 0
    test =  brier(fcounts, obs, fcst_percent, obs_percent)
    ans = 0.152
    print(test, ans)
    
    fcounts = np.array([2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0])
    obs = np.array([0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0])
    obs_percent = obs/fcounts
    obs_percent[np.isnan(obs_percent)] = 0
    test =  brier(fcounts, obs, fcst_percent, obs_percent)
    ans = 0.160
    print(test, ans)

    fcounts = np.array([0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0])
    obs = np.array([0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0])
    obs_percent = obs/fcounts
    obs_percent[np.isnan(obs_percent)] = 0
    test =  brier(fcounts, obs, fcst_percent, obs_percent)
    ans = 0.240
    print(test, ans)
    
    fcounts = np.array([6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    obs = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3])
    obs_percent = obs/fcounts
    obs_percent[np.isnan(obs_percent)] = 0
    test =  brier(fcounts, obs, fcst_percent, obs_percent)
    ans = 0.2
    print(test, ans)





if __name__ == '__main__':
    test_brier()
