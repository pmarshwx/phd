import os
import hwt
import pygrib
import numpy as np
from scipy import ndimage
from disser import stat_tools
import disser.misc
import metpy as mp


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
                stg4_thresh : number (default 25.4)
                The observed precipitation value being verified against.
            stg4_thresh : number (default 25.4)
                The observed precipitation value used as threshold
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
        stg4 = np.ma.asanyarray(stg4).filled(missing)
        fcst = pygrib.open(fcst_file)[1]['values']
        fcst = np.ma.asanyarray(fcst).filled(missing)
        if isinstance(mask, type(None)):
            mask = np.ones(stg4.shape, dtype='int')
        # Only compare grid points that have valid data
        # in both forecast and observations
        stg4_exceed, fcst_exceed = hwt.neighborhood.find_joint_exceed(
                stg4, fcst, mask, stg4_thresh, fcst_thresh, missing)
        # Create the histogram
        hist2d = hwt.neighborhood.error_composite(
                fcst_exceed.astype(int), stg4_exceed.astype(int), radius, dx)
        hist2d[hist2d<0] = -1
        hist2d = hist2d.reshape(nx, nx)
        np.savez_compressed(nout_file, hist2d=hist2d, stg4_thresh=stg4_thresh,
                            fcst_thresh=fcst_thresh, amts=amts, dx=dx,
                            radius=radius)


def make_frequency_list(kwargs):
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
            convert_factor : integer (default 8)
                Number needed to multiply data by to convert
                from decimal to integer
            max_precip_mm : number (default 400)
                The maximum precipitation value. All precip values
                greater than this are considered to be this value.
                stg4_thresh : number (default 25.4)
                The observed precipitation value being verified against.
            missing : number (default -9999)
                The value of missing data in either the fcst or stg4 files.
            mask : 2D Array
                2D mask (1/0) with 1 being good 0 being masked

    Returns
    -------
    None

    """
    convert_factor = kwargs.get('convert_factor', 8)
    max_precip_mm = kwargs.get('max_precip_mm', 400)
    missing = kwargs.get('missing', -9999)
    # Extract Files
    stg4_files = kwargs.get('stg4_files', None)
    fcst_files = kwargs.get('fcst_files', None)
    nout_files = kwargs.get('nout_files', None)
    mask = kwargs.get('mask', None)
    files = zip(stg4_files, fcst_files, nout_files)
    amts = np.arange(max_precip_mm * convert_factor + 1).astype(float)
    amts /= float(convert_factor)
    amts_len = amts.shape[0]
    # Loop through files and create histograms
    for stg4_file, fcst_file, nout_file in files:
        if not disser.misc.fsize_check(stg4_file): continue
        if not disser.misc.fsize_check(fcst_file): continue
        stg4 = pygrib.open(stg4_file)[1]['values']
        stg4 = np.ma.asanyarray(stg4).filled(missing)
        fcst = pygrib.open(fcst_file)[1]['values']
        fcst = np.ma.asanyarray(fcst).filled(missing)
        if isinstance(mask, type(None)):
            mask = np.ones(stg4.shape, dtype='int')
        stg4_int = (stg4 * convert_factor).astype(int)
        fcst_int = (fcst * convert_factor).astype(int)
        stg4_dist, fcst_dist = hwt.bin.joint_precip(stg4_int, fcst_int,
                                                    mask, amts_len)
        np.savez_compressed(nout_file, stg4_dist=stg4_dist,
                            fcst_dist=fcst_dist, amts=amts)


def get_simulation_params(kwargs):
    """
    Fits a 2D anisotropic Gaussian to a 2D frequency diagram

    Parameters
    ----------
    kwargs : dict

        Mandatory Keywords:
            sim_number : array_like
                The current simulation number
            dts : array_like
                List of valid training dates
            npbin_root : string
                The root directory for which the forecasts will be stored
            members : array_like
                An array_like object with strings of members' names to use

    Returns
    -------
    Tuple
        The simulation number and a dictionary with the
        fitted distribution parameters as keywords

    """
    sim_number = kwargs.get('sim_number')
    dts = kwargs.get('dts')
    npbin_root = kwargs.get('npbin_root')
    members = kwargs.get('members')
    include_year = kwargs.get('include_year', True)
    parms = {}
    for member in members:
        init = True
        for dt in dts:
            date4 = dt.strftime('%Y%m%d%H')
            year = date4[:4]
            npbin_path = os.path.join(npbin_root, member)
            if include_year:
                npbin_path = os.path.join(npbin_path, year)
            npbin_file = os.path.join(npbin_path, '%s.npz' % (date4))
            f = np.load(npbin_file)
            if init:
                hist = f['hist2d']
                stg4_thresh = f['stg4_thresh']
                fcst_thresh = f['fcst_thresh']
                init = False
            else:
                hist += f['hist2d']
            f.close()
        hist = np.ma.asarray(hist)
        hist[hist < 0] = np.ma.masked
        sigx, sigy, xrot = mp.kde.fit_gauss2d_alt(hist)
        nx = hist.shape[0]
        hnx = np.floor(nx / 2)
        y, x = ndimage.center_of_mass(hist)
        x -= hnx
        y -= hnx
        ih = round(x)
        ik = round(y)
        if x > (-0.5 + 1e-11) and x < 0:
            ih = -ih
        if y > (-0.5 + 1e-11) and y < 0:
            ik = -ik
        parms[member] = {
                'sigx': sigx, 'sigy': sigy, 'xrot': xrot, 'h': x, 'k': y,
                'fcst_thresh': fcst_thresh, 'stg4_thresh': stg4_thresh,
                'ih': ih, 'ik': ik}
    return sim_number, parms


