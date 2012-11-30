import pygrib
import numpy as np
import hwt
import disser.misc


def _create_forecast(kwargs):
    """
    A thin wrapper around the create_forecast function. Designed to give
    access to the actual forecast creation routine via a dictionary of
    parameters instead of direct keywords.

    Parameters
    ----------
    kwargs : dict

        Mandatory Keywords:
            stg4 : array_like
                Actual Forecast data (binary 1/0 grid)
            fcst : array_like
                Actual forecast data
            nout_file : string
                Output file
            sigx : float
                The larger sigma value
            sigy : float
                The smaller sigma value
            mask : 2D Array
                2D mask (1/0) with 1 being good 0 being masked

        Optional Keywords:
            stg4_thresh : float (default 25.4)
                Threshold which to make the forecast from
            fcst_thresh : float (default 25.4)
                Threshold which to make the forecast from
            dx : float (default 4.7)
                The grid spacing of the forecast data
            xrot : float (default 0)
                The angle by which the x-axis must be rotated to match the
                orientation of the larger sigma value
            h : int (default 0)
                The x-distance of the offset
            k : int (default 0)
                The y-distance of the offset
            factor : float (default 3)
                The number of standard deviations to include in calculation
            thresh : float (default 25.4)
                Threshold which to make the forecast from
    """
    nout_file = kwargs.get('nout_file')
    stg4 = kwargs.get('stg4')
    fcst = kwargs.get('fcst')
    mask = kwargs.get('mask')
    sigx = kwargs.get('sigx')
    sigy = kwargs.get('sigy')
    xrot = kwargs.get('xrot', 0.)
    h = kwargs.get('h', 0)
    k = kwargs.get('k', 0)
    dx = kwargs.get('dx', 4.7)
    factor = kwargs.get('factor', 3)
    stg4_thresh = kwargs.get('stg4_thresh', 25.4)
    fcst_thresh = kwargs.get('fcst_thresh', 25.4)
    create_forecast(nout_file, stg4, fcst, mask, sigx, sigy, xrot,
                    h, k, dx, factor, stg4_thresh, fcst_thresh)


def create_forecast(nout_file, stg4, fcst, mask, sigx, sigy, xrot,
                    h, k, dx, factor, stg4_thresh, fcst_thresh):
    """
    A thin wrapper around the create_forecast function. Designed to give
    access to the actual forecast creation routine via a dictionary of
    parameters instead of direct keywords.

    Parameters
    ----------
    stg4 : array_like
        Actual Forecast data (binary 1/0 grid)
    fcst : array_like
        Actual forecast data
    nout_file : string
        Output file
    sigx : float
        The larger sigma value
    sigy : float
        The smaller sigma value
    mask : 2D Array
        2D mask (1/0) with 1 being good 0 being masked
    stg4_thresh : float (default 25.4)
        Threshold which to make the forecast from
    fcst_thresh : float (default 25.4)
        Threshold which to make the forecast from
    dx : float (default 4.7)
        The grid spacing of the forecast data
    xrot : float (default 0)
        The angle by which the x-axis must be rotated to match the
        orientation of the larger sigma value
    h : int (default 0)
        The x-distance of the offset
    k : int (default 0)
        The y-distance of the offset
    factor : float (default 3)
        The number of standard deviations to include in calculation
    stg4_thresh : float (default 25.4)
        Threshold which to make the forecast from
    fcst_thresh : float (default 25.4)
        Threshold which to make the forecast from

    """
    fcst_aniso = hwt.smoothers.anisotropic_gauss(
            fcst, sigx, sigy, xrot, h, k, dx, factor, True)
    fcst_aniso *= 100
    stg4[mask] = -9999
    np.savez_compressed(
        nout_file, fcst=fcst, stg4=stg4, fcst_aniso=fcst_aniso,
        stg4_thresh=stg4_thresh, fcst_thresh=fcst_thresh,
        factor=factor, sigx=sigx, sigy=sigy, xrot=xrot, h=h, k=k, dx=dx)


def create_forecasts(kwargs):
    """
    Create a probabilistic forecast using kernel density estimation
    from a 2D Gaussian fit to the 2D histograms of where observed precipitation
    occurred relative to forecast.

    Parameters
    ----------
    kwargs : dict

        Mandatory Keywords:
            stg4_files : array_like
                List of observation files
            fcst_files : array_like
                List of forecast files
            nout_files : array_like
                List of output files
            sigx : str, float
                The larger sigma value
            sigy : str, float
                The smaller sigma value
            mask : 2D Array
                2D mask (1/0) with 1 being good 0 being masked

        Optional Keywords:
            thresh : str, float (default 25.4)
                Threshold which to make the forecast from
            stg4_thresh : str, float (default 25.4)
                Threshold which to make the forecast from
            fcst_thresh : str, float (default 25.4)
                Threshold which to make the forecast from
            dx : str, float (default 4.7)
                The grid spacing of the forecast data
            xrot : str, float (default 0)
                The angle by which the x-axis must be rotated to match the
                orientation of the larger sigma value
            h : str, int (default 0)
                The x-distance of the offset
            k : str, int (default 0)
                The y-distance of the offset
            factor : str, float (default 3)
                The number of standard deviations to include in calculation

    Returns
    -------
    None

    """
    # Parse Keyword Dictionary
    stg4_files = kwargs.get('stg4_files', None)
    fcst_files = kwargs.get('fcst_files', None)
    nout_files = kwargs.get('nout_files', None)
    sigx = kwargs.get('sigx', None)
    if not sigx: sigx = float(sigx)
    sigy = kwargs.get('sigy', None)
    if not sigy: sigy = float(sigy)
    mask = kwargs.get('mask', None)
    if not stg4_files or not fcst_files or not nout_files or \
            not sigx or not sigy:
        raise Exception("Must Include Input/Output Files")
    else:
        files = zip(stg4_files, fcst_files, nout_files)
    dx = float(kwargs.get('dx', 4.7))
    xrot = float(kwargs.get('xrot', 0.))
    h = int(kwargs.get('h', 0))
    k = int(kwargs.get('k', 0))
    factor = float(kwargs.get('factor', 3.))
    thresh = float(kwargs.get('thresh', 25.4))
    stg4_thresh = float(kwargs.get('stg4_thresh', stg4_thresh))
    fcst_thresh = float(kwargs.get('fcst_thresh', fcst_thresh))
    # Create Forecast
    for stg4_file, fcst_file, nout_file in files:
        if not disser.misc.fsize_check(stg4_file): continue
        if not disser.misc.fsize_check(fcst_file): continue
        stg4_grb = pygrib.open(stg4_file)
        stg4 = stg4_grb[1]['values']
        stg4_grb.close()
        fcst_grb = pygrib.open(fcst_file)
        fcst = fcst_grb[1]['values']
        fcst_grb.close()
        if isinstance(mask, type(None)):
            mask = np.ones(stg4.shape, dtype='int')
        stg4_d, fcst_d = hwt.neighborhood.find_joint_exceed(
                stg4, fcst, mask, stg4_thresh, fcst_thresh)
        create_forecast(nout_file, stg4_d, fcst_d, stg4.mask,
                        sigx, sigy, xrot, h, k, dx, factor,
                        stg4_thresh, fcst_thresh)


def forecast_verification(kwargs):
    """
    A script to create the verification files.

    Mandatory Keywords:
        fcst_files : array_like
            List of forecast files
        verif_file : str
            String containing the full path to the verification file
            to be written

    Optional Keywords:
        field : str (default 'fcst_aniso')
            Variable name of the file in the numpy binary file
        thresh : number (default 25.4)
            Threshold at which to conduce the exceedance forecast
        precision : integer (default 0)
            The number of decimal points used in the forecast probabilities
        missing : integer (default -9999)
            The value to treat as missing data
        mask : 2D Array
            2D mask (1/0) with 1 being good 0 being masked

    """
    fcst_files = kwargs.get('fcst_files', None)
    verif_file = kwargs.get('verif_file', None)
    if not fcst_files:
        raise Exception("Needed Files Not Present")
    field = kwargs.get('field', 'fcst_aniso')
    thresh = float(kwargs.get('thresh', 25.4))
    precision = int(kwargs.get('precision', 0))
    missing = int(kwargs.get('missing', -9999))
    mask = kwargs.get('mask', None)
    initial = True
    multfactor = 10**precision
    for fcst_file in fcst_files:
        f = np.load(fcst_file)
        fcst_prob = f[field]
        stg4 = f['stg4']
        try:
            fcst = f['fcst']
        except:
            fcst = None
        f.close()
        if isinstance(mask, type(None)):
            mask = np.ones(stg4.shape, dtype='int')
        fcst_prob_v = fcst_prob.copy()
        fcst_prob_v *= multfactor
        fcst_prob_v[fcst_prob_v == 0] = -1
        fhist, ohist = hwt.verification.reliability(
                fcst_prob_v.astype('int'), stg4.astype('int'), mask,
                100*multfactor, missing)
        stg4 = np.ma.asarray(stg4)
        stg4[stg4 == missing] = np.ma.masked
        if initial:
            stg4_total = stg4.sum()
            if not isinstance(fcst, type(None)):
                fcst_total = fcst.sum()
            else:
                fcst_total = missing
            ftotal = fhist.copy()
            ototal = ohist.copy()
            initial = False
        else:
            stg4_total += stg4.sum()
            if not isinstance(fcst, type(None)):
                fcst_total += fcst.sum()
            else:
                fcst_total = missing
            ftotal += fhist
            ototal += ohist
    if verif_file:
        bias = fcst_total/stg4_total
        if np.abs(bias - missing) < 1:
            bias = missing
        write_file(verif_file, fcst_total, stg4_total, bias,
                   ftotal, ototal, precision)
    else:
        return fcst_total, stg4_total, ftotal, ototal


def write_file(out, nssl, stg4, bias, ftotal, ototal, precision):
    """
    A function to create the CSV file of contingency tables.

    Paramters
    ---------
    out : str
        The full path (including file name) of the file to which
        the data are written
    nssl : int
        The total number of grid point "Yes" forecasts in the forecast
    stg4 : int
        The total number of grid point "Yes" observations
    bias : float
        The bias of forecast / observations
    ftotal : 1d numpy array
        The number of grid point "Yes" forecasts at each probability threshold
    ototal : 1d numpy array
        The number of grid point "Yes" observations at each
        probability threshold

    Returns
    -------
    None

    """
    fout = open(out, 'w')
    fout.write('### NSSL TOTAL: %i\n' % (nssl))
    fout.write('### STAGE IV TOTAL: %i\n' % (stg4))
    fout.write('### BIAS ###: %.4f\n' % (bias))
    fout.write('fpercent, opercent, ocount, fcount, fptotal, optotal\n')
    fsum = ftotal.sum().astype('float')
    osum = ototal.sum().astype('float')
    for i in range(len(ftotal)):
        fcount = ftotal[i]
        ocount = ototal[i]
        fpercent = (i-1) / 10**precision
        try:
            opercent = ocount/fcount * 100
        except ZeroDivisionError:
            if ocount == 0:
                opercent = 0
            else:
                opercent = 101
        if np.isnan(opercent):
            opercent = 0.
        fptotal = fcount/fsum * 100
        optotal = ocount/osum * 100
        if np.isnan(optotal):
            optotal = 0
        fout.write('%.2f, %.2f, %i, %i, %.2f, %.2f\n' % (fpercent, opercent,
                   ocount, fcount, fptotal, optotal))
    fout.close()

