"""
A collection of conversion routines
"""
from __future__ import print_function, division
import numpy as np


__all__ = ['reliability2contingency']


def reliability2contingency(fcsts, obs):
    """ 
    A function to convert reliability table into a Contingency Table.
    
    This routine assumes that fcsts and obs are two ordered arrays where
    the same elements of fcsts and obs correspond to the same forecast
    threshold. 
    
    Parameters
    ----------
    fcsts : list or numpy array               
        Array of the number of forecasts at a given forecast percent
    obs : list or numpy array
        Array of the number of observed grid points at a given 
        forecast threshold

    Returns
    -------
    a : numpy array
        An array of floats containing the 'a' values of a contingency
        table. 
    b : numpy array
        An array of floats containing the 'b' values of a contingency
        table. 
    c : numpy array
        An array of floats containing the 'c' values of a contingency
        table. 
    d : numpy array
        An array of floats containing the 'd' values of a contingency
        table. 
    """
    fcsts = np.array(fcsts)
    obs = np.array(obs)

    assert len(fcsts.shape) == 1, 'fcst must be 1-d'
    assert len(obs.shape) == 1, 'obs must be 1-d'
    
    a = np.zeros_like(fcsts)
    b = np.zeros_like(fcsts)
    c = np.zeros_like(fcsts)
    d = np.zeros_like(fcsts)
    
    fsum = fcsts.sum()
    osum = obs.sum()
    for i in range(fcsts.shape[0]):
        a[i] = obs[i:].sum()
        b[i] = fcsts[i:].sum() - a[i]
        c[i] = osum - a[i]
        d[i] = fsum - fcsts[i:].sum() - osum + a[i]
    
    return (a, b, c, d)
    
    
def test_reliability2contingency():
    """ 
    Test Suite of the reliability2contingency function.
    """
    import matplotlib.pyplot as plt
    import veripy.contingency as contingency
    
    fcsts = np.array([100., 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
    obs = np.array([10., 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    obs = np.ones_like(fcsts)
    obs = np.minimum(obs, fcsts)

    a, b, c, d = reliability2contingency(fcsts, obs)
    pod = contingency.pod(a, b, c, d)
    pofd = contingency.pofd(a, b, c, d)
    
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.plot(pofd, pod)
    diag = np.linspace(0, 1)
    ax.plot(diag, diag, color='black', linestyle='dashed', linewidth=0.5)
    ax.set_yticks(np.arange(0, 1+1e-6, .1))
    ax.set_xticks(ax.get_yticks())
    ax.set_aspect('equal')
    ax.grid()
    ax.set_title('Area Under Curve: %.4f' % (-1 * np.trapz(pod, pofd)))
    plt.show()





if __name__ == "__main__":
    test_reliability2contingency()
    
    
    
    
