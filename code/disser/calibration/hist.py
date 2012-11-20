import hwt
import pygrib
import numpy as np
from disser import stat_tools
import disser.misc


def create_hist2d(kwargs):
    """
    Create a 2D histogram of where observed precipitation
    occurred relative to forecast.

    Parameters
    ----------
    kwargs : dict

        Mandatory Keywords:
            stg4_files : array_like
                List of observations files
            fcst_files : array_like
                List of forecast files
            nout_files : array_like
                List of output files

        Optional Keywords:
            radius : number (default 400)
                Radius of circle used for 2D Histogram
            dx : number (default 4.7)
                Distance between grid points
            convert_factor : integer (default 8)
                Number needed to multiply data by to convert
                from decimal to integer
            max_precip_mm : number (default 400)
                The maximum precipitation value. All precip values
                greater than this are considered to be this value.
            stg4_quantile : number (default 99.9)
                Quantile of precipitation values used for quantile analysis.
                If thresholds are set, quantiles are ignored.
            fcst_quantile : number (default 99.9)
                Quantile of precipitation values used for quantile analysis.
                If thresholds are set, quantiles are ignored.
            stg4_thresh : number (default 25.4)
                The observed precipitation value being verified against.
            fcst_thresh : number (default 25.4)
                The forecast precipitation value used as threshold
            missing : number (default -9999)
                The value of missing data in either the fcst or stg4 files.
            mask : 2D Array
                2D mask (1/0) with 1 being good 0 being masked

    Returns
    -------
    None

    """
    radius = kwargs.get('radius', 400)
    dx = kwargs.get('dx', 4.7)
    convert_factor = kwargs.get('convert_factor', 8)
    max_precip_mm = kwargs.get('max_precip_mm', 400)
    hnx = int(radius / dx)
    nx = hnx * 2 + 1
    # Configurations for Quantile Processing
    fcst_quantile = kwargs.get('fcst_quantile', 99.9)
    stg4_quantile = kwargs.get('stg4_quantile', 99.9)
    min_stg4_thresh = kwargs.get('min_stg4_thresh', 25.4)
    missing = kwargs.get('missing', -9999)
    # Configurations for Exact Amount Processing
    fcst_thresh = kwargs.get('fcst_thresh', 25.4)
    stg4_thresh = kwargs.get('stg4_thresh', 25.4)
    # Extract Files
    stg4_files = kwargs.get('stg4_files', None)
    fcst_files = kwargs.get('fcst_files', None)
    nout_files = kwargs.get('nout_files', None)
    mask = kwargs.get('mask', None)
    files = zip(stg4_files, fcst_files, nout_files)
    amts = np.arange(max_precip_mm * convert_factor + 1) / convert_factor
    amts_len = amts.shape[0]
    # Loop through files and create histograms
    for stg4_file, fcst_file, nout_file in files:
        if not disser.misc.fsize_check(stg4_file): continue
        if not disser.misc.fsize_check(fcst_file): continue
        stg4 = pygrib.open(stg4_file)[1]['values']
        stg4 = np.ma.asanyarray(stg4).filled(-9999)
        fcst = pygrib.open(fcst_file)[1]['values']
        fcst = np.ma.asanyarray(fcst).filled(-9999)
        if isinstance(mask, type(None)):
            mask = np.ones(stg4.shape, dtype='int')
        stg4_int = (stg4 * convert_factor).astype(int)
        fcst_int = (fcst * convert_factor).astype(int)
        stg4_dist, fcst_dist = hwt.bin.joint_precip(stg4_int, fcst_int,
                                                    mask, amts_len)
        if not stg4_thresh and not fcst_thresh:
            stg4_thresh = stat_tools.quantile_to_value(
                    amts, stg4_dist, stg4_quantile)
            fcst_thresh = stat_tools.quantile_to_value(
                    amts, fcst_dist, fcst_quantile)
            if stg4_thresh < min_stg4_thresh: continue
        elif stg4_thresh and not fcst_thresh:
            stg4_quantile = stat_tools.value_to_quantile(
                    amts, stg4_dist, stg4_thresh)
            fcst_thresh = stat_tools.quantile_to_value(
                    amts, fcst_dist, stg4_quantile)
            fcst_quantile = stat_tools.value_to_quantile(
                    amts, stg4_dist, fcst_thresh)
        # Only compare grid points that have valid data
        # in both forecast and observations
        stg4_exceed, fcst_exceed = hwt.neighborhood.find_joint_exceed(
                stg4, fcst, mask, stg4_thresh, fcst_thresh, missing)
        # Create the histogram
        hist2d = hwt.neighborhood.error_composite(
                fcst_exceed.astype(int), stg4_exceed.astype(int), radius, dx)
        hist2d[hist2d<0] = -1
        hist2d = hist2d.reshape(nx, nx)
        np.savez_compressed(nout_file, hist2d=hist2d, stg4_dist=stg4_dist,
                            fcst_dist=fcst_dist, stg4_thresh=stg4_thresh,
                            fcst_thresh=fcst_thresh, amts=amts, dx=dx,
                            fcst_quantile=fcst_quantile, radius=radius,
                            stg4_quantile=stg4_quantile)
