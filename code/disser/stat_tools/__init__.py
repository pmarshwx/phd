import numpy as np
import scipy.stats as spstats
import statsmodels as smodels


def make_ecdf(counts):
    '''
    This function takes an ordered list of counts (similar to what is
    returned by numpy's histogram function, and returns an emperical
    cumulative distribution function.

    Parameters
    ----------
    counts : array_like
        An ordered sequence of counts

    Returns
    -------
    The emperical cumulative distribution function

    '''
    cdf = np.cumsum(counts)
    cdf /= float(cdf[-1])
    return cdf


def quantile_to_value(values, ecdf, quantile):
    '''
    This function takes an emperical cumulative distribution function and
    values as separate arrays and returns the threshold based on the supplied
    quantile.

    Parameters
    ----------
    values : array_like
        The values
    ecdf : array_like
        The emperical cumulative distribution function
    quantile : float
        The quantile being evaluated

    Returns
    -------
    The value of the user supplied quantile.

    '''
    ecdf = np.asarray(ecdf)
    values = np.asarray(values)
    return values[ecdf >= quantile][0]


def value_to_quantile(values, ecdf, value):
    '''
    This function takes an emperical cumulative distribution function and
    values as separate arrays and returns the quantile based on the supplied
    value.

    Parameters
    ----------
    values : array_like
        The values
    ecdf : array_like
        Counts of each value
    value : number
        The value of the desired quantile

    Returns
    -------
    The quantile of the supplied threshold

    '''
    ecdf = np.asarray(ecdf)
    values = np.asarray(values)
    return ecdf[np.where(values >= value)[0][0]]


