"""
A collection of various meteorological verification skill scores.
"""
from __future__ import print_function, division
import numpy as np
from veripy import components, scores





def brier(bs, bs_ref):
    """
    Compute the Brier Skill Score
    
    User supplies the Brier Score for the forecast in question as well 
    as a reference forecast, and the Brier Skill Score is returned.  
    
    Note
    ----
    A special version of this function exists to compute the Brier Skill
    Score for situations where the reference forecast is climatology.
    
    Parameters
    ----------
    bs : float
        Brier Score for forecasts being scored
    bs_ref : float
        Brier Score for reference forecast
        
    Returns
    -------
    bss : float
        The Brier Skill Score
    """
    
    return (1 - (bs/bs_ref))
    

def brier_climo(fcsts, obs, fcst_percent, obs_percent):
    """
    A alternative function to compute the Brier Skill Score.

    Notes
    -----
    A special version of 'brier' used to compute the Brier Skill Score
    where the reference forecast is taken to be climatology

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
    brier__skill_score : float
        The Brier Skill Score
    """

    reliability = components.get_reliability(fcsts, fcst_percent, obs_percent)
    resolution = components.get_resolution(fcsts, obs, obs_percent)
    uncertainty = components.get_uncertainty(fcsts, obs)

    return (resolution - reliability) / uncertainty





if __name__ == '__main__':
    pass