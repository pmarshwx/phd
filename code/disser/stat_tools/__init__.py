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
    cdf /= cdf[-1]
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


def value_to_quantile(values, counts, value):
    '''
    This function takes a frequency table via the frequencies and counts as
    separate arrays and returns the quantile based on the supplied frequency.

    Parameters
    ----------
    values : array_like
        The values
    counts : array_like
        Counts of each value
    value : number
        The value of the desired quantile

    Returns
    -------
    quantile : float
        The quantile of the supplied threshold

    '''
    values = np.asanyarray(values)
    counts = np.asanyarray(counts)
    value = np.float(value)
    error = [(a - value) for a in values]
    abs_error = np.abs(error)
    min_abs_error = np.min(abs_error)
    ind = np.where(abs_error == min_abs_error)[0][0]
    error_amount = error[ind]
    # Because we are dealing with exceedances, if the error is less than zero,
    # it means the nearest frequency actually exceeds supplied threshold.
    # Thus we need to take the next highest frequency, even though the error
    # will be slightly more.
    if error_amount < 0:
        ind += 1
        if ind >= values.shape[0]:
            ind = values.shape[0]-1
    quantile_array = counts.cumsum() / counts.sum()
    quantile = quantile_array[ind] * 100
    return quantile