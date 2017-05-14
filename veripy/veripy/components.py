"""
Components of various verification scores and verification skill scores.
"""
from __future__ import print_function, division
import numpy as np


__all__ = ['get_reliability', 'get_resolution', 'get_uncertainty']


def get_reliability(fcsts, fcst_percent, obs_percent):
    """ 
    A function to compute the reliability component of the Brier Score.
    
    
    Parameters
    ----------
    fcsts : numpy array               
        Array of the number of forecasts at a given forecast percent
    fcst_percent : numpy array
        An ordered array of the forecast probabilities.
    obs_percent : numpy array
        The ratio of observation points to forecast points at the
        corresponding forecast percent (fcst_percent)

    Returns
    -------
    reliability : float
        Reliability component of the Brier Score 
    """
    
    return (fcsts * (fcst_percent - obs_percent)**2).sum() / fcsts.sum()


def get_resolution(fcsts, obs, obs_percent):
    """
    A function to compute the resolution component of the Brier Score.
    
    Parameters
    ----------
    fcsts : numpy array               
        Array of the number of forecasts at a given forecast percent
    obs : numpy array
        Array of the number of observed grid points at a given 
        forecast threshold
    obs_percent : numpy array
        The ratio of observation points to forecast points at the
        corresponding forecast percent (fcst_percent)

    Returns
    -------
    resolution : float
        Resolution component of the Brier Score
    """
    
    fcst_sum = fcsts.sum()
    climo = obs.sum() / fcst_sum
    return (fcsts * (obs_percent - climo)**2).sum() / fcst_sum
    
    
def get_uncertainty(fcsts, obs):
    """
    A function to compute the uncertainty component of the Brier Score.
    
    Parameters
    ----------
    fcsts : numpy array               
        Array of the number of forecasts at a given forecast percent
    obs : numpy array
        Number of observed grid points at a given forecast threshold
                                            
    Returns
    -------
    uncertainty : float
        Uncertainty component of the Brier Score
    """
    
    climo = obs.sum() / fcsts.sum()
    return (climo * (1 - climo))

    
    
    
        
if __name__ == '__main__':
    pass    
    
    
    
    
    
    