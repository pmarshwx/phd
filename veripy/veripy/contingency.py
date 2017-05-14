"""
A collection of verification metrics derived from a contingency table.
"""
from __future__ import print_function, division
import numpy as np

import veripy.convert as convert

__all__ = ['pod', 'pofd', 'far', 'csi', 'bias']


def pod(a, b, c, d):
    """
    Compute the Probability of Detection (PoD)
    
    Parameters
    ----------
    a : list, tuple, or numpy array
        Array of values containing the 'a' component of a contingency
        table for each forecast threshold
    b : list, tuple, or numpy array
        Array of values containing the 'b' component of a contingency
        table for each forecast threshold
    c : list, tuple, or numpy array
        Array of values containing the 'c' component of a contingency
        table for each forecast threshold
    d : list, tuple, or numpy array
        Array of values containing the 'd' component of a contingency
        table for each forecast threshold
    
    Returns
    -------
    _pod : numpy array
        Array of floats containing the Probability of Detection at 
        each forecast threshold.
    """
    a = np.array(a)
    c = np.array(c)
    _pod = a / (a + c)
    _pod[np.isnan(_pod)] = 0
    return _pod


def pofd(a, b, c, d):
    """
    Compute the Probability of False Detection (PoFD)
    
    Parameters
    ----------
    a : list, tuple, or numpy array
        Array of values containing the 'a' component of a contingency
        table for each forecast threshold
    b : list, tuple, or numpy array
        Array of values containing the 'b' component of a contingency
        table for each forecast threshold
    c : list, tuple, or numpy array
        Array of values containing the 'c' component of a contingency
        table for each forecast threshold
    d : list, tuple, or numpy array
        Array of values containing the 'd' component of a contingency
        table for each forecast threshold
    
    Returns
    -------
    _pofd : numpy array
        Array of floats containing the Probability of False Detection at 
        each forecast threshold.
    """
    b = np.array(b)
    d = np.array(d)
    _pofd = b / (b + d)
    _pofd[np.isnan(_pofd)] = 0
    return _pofd


def far(a, b, c, d):
    """
    Compute the False Alarm Ratio (FAR)
    
    Parameters
    ----------
    a : list, tuple, or numpy array
        Array of values containing the 'a' component of a contingency
        table for each forecast threshold
    b : list, tuple, or numpy array
        Array of values containing the 'b' component of a contingency
        table for each forecast threshold
    c : list, tuple, or numpy array
        Array of values containing the 'c' component of a contingency
        table for each forecast threshold
    d : list, tuple, or numpy array
        Array of values containing the 'd' component of a contingency
        table for each forecast threshold
    
    Returns
    -------
    _far : numpy array
        Array of floats containing the False Alarm Ratio at 
        each forecast threshold.
    """
    a = np.array(a)
    b = np.array(b)
    _far = b / (a + b)
    _far[np.isnan(_far)] = 0
    return _far

    
def csi(a, b, c, d):
    """
    Compute the Critical Success Ratio (CSI)
    
    Parameters
    ----------
    a : list, tuple, or numpy array
        Array of values containing the 'a' component of a contingency
        table for each forecast threshold
    b : list, tuple, or numpy array
        Array of values containing the 'b' component of a contingency
        table for each forecast threshold
    c : list, tuple, or numpy array
        Array of values containing the 'c' component of a contingency
        table for each forecast threshold
    d : list, tuple, or numpy array
        Array of values containing the 'd' component of a contingency
        table for each forecast threshold
    
    Returns
    -------
    _csi : numpy array
        Array of floats containing the Critical Success Index at 
        each forecast threshold.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    _csi = a / (a + b + c)
    _csi[np.isnan(_csi)] = 0
    return _csi


def bias(a, b, c, d):
    """
    Compute the Bias.
    
    Parameters
    ----------
    a : list, tuple, or numpy array
        Array of values containing the 'a' component of a contingency
        table for each forecast threshold
    b : list, tuple, or numpy array
        Array of values containing the 'b' component of a contingency
        table for each forecast threshold
    c : list, tuple, or numpy array
        Array of values containing the 'c' component of a contingency
        table for each forecast threshold
    d : list, tuple, or numpy array
        Array of values containing the 'd' component of a contingency
        table for each forecast threshold
    
    Returns
    -------
    _bias : numpy array
        Array of floats containing the Bias at each forecast threshold.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    _bias = (a + b) / (a + c)
    _bias[np.isnan(_bias)] = 0
    return _bias


def test():
    """ 
    Test suite for computing verification metrics from contingency tables.
    """
    fcsts = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    obs = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    a, b, c, d = convert.reliability2contingency(fcsts, obs)
    _pod = pod(a, b, c, d)
    _pofd = pofd(a, b, c, d)
    _far = far(a, b, c, d)
    _csi = csi(a, b, c, d)
    _bias = bias(a, b, c, d)
    print(_pod)
    print(_pofd)
    print(_far)
    print(_csi)
    print(_bias)





if __name__ == '__main__':
    test()   