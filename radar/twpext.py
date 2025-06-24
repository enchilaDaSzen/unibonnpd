"""
Towerpy extension & add-ons.

Some of these utilities may be added in following updates of towerpy.
# =============================================================================

@author: dsanche1@uni-bonn.de
"""

import copy
import time
import os
import datetime as dt
import numpy as np
from scipy import constants as sc
import wradlib as wrl
from zoneinfo import ZoneInfo
from towerpy.utils import radutilities as rut
from towerpy.georad import georef_rdata as geo
from towerpy.utils import unit_conversion as tpuc
from towerpy.datavis import rad_display
from towerpy.eclass.snr import SNR_Classif
# from towerpy.utils.radutilities import rolling_window
from towerpy.utils.radutilities import linspace_step
import warnings
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mpc
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import netCDF4 as nc
from itertools import zip_longest
from sklearn.metrics import DistanceMetric
import scipy.spatial as spatial
from matplotlib.patches import Rectangle
import csv


test_pckgs = {'towerpy': '1.0.7', 'wradlib': '2.4.0', 'scipy': '1.15.1',
              'cartopy': '0.24.0', 'scikit-learn': '1.6.1', 'numpy': '1.26.4',
              'python': '3.10.12',  'matplotlib': '3.9.1'}
latest_update = dt.datetime(2025, 1, 31, 17, 0)
print_rtimes = False


# =============================================================================
# utils
# =============================================================================
def bbox(*args):
    """Get bounding box from a set of bin coordinates."""
    x = np.array([])
    y = np.array([])
    for arg in args:
        x = np.append(x, arg[:, 0])
        y = np.append(y, arg[:, 1])
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    return xmin, xmax, ymin, ymax


def time_mod(time, delta, epoch=None):
    """
    Compute datetime as POSIX time.

    Parameters
    ----------
    time : datetime object
        DESCRIPTION.
    delta : datetime object
        DESCRIPTION.
    epoch : datetime object, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    time = time.replace(tzinfo=None)
    if epoch is None:
        epoch = dt.datetime(1970, 1, 1, tzinfo=time.tzinfo)
    return (time - epoch) % delta


def time_round(time, delta, epoch=None):
    """
    Round given datetime to nearest datetime.

    Parameters
    ----------
    time : datetime object
        DESCRIPTION.
    delta : datetime object
        DESCRIPTION.
    epoch : datetime object, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mod = time_mod(time, delta, epoch)
    if mod < delta / 2:
        return time - mod
    return time + (delta - mod)


def time_floor(time, delta, epoch=None):
    """
    Compute the floor of given datetime to nearest datetime.

    Parameters
    ----------
    time : datetime object
        DESCRIPTION.
    delta : datetime object
        DESCRIPTION.
    epoch : datetime object, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mod = time_mod(time, delta, epoch)
    return time - mod


def time_ceil(time, delta, epoch=None):
    """
    Compute the ceiling value of given datetime.

    Parameters
    ----------
    time : datetime object
        DESCRIPTION.
    delta : datetime object
        DESCRIPTION.
    epoch : datetime object, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mod = time_mod(time, delta, epoch)
    if mod:
        return time + (delta - mod)
    return time


def fill_timeseries(dtso, valso, stspdt=None, deltadt=dt.timedelta(minutes=5),
                    dtround=dt.timedelta(minutes=1), fillv=np.nan, tz=None,
                    toldt=dt.timedelta(minutes=1), return_index=False):
    """
    Fill datetime series.

    Parameters
    ----------
    dtso : list
        List containing date/time objects. First and last objects are used to
        set the start and stop time, respectively.
    valso : array
        Values corresponding to the timesteps.
    stspdt : 2-element list or tuple, optional
        Overraids the start/stop date/time. The default is None.
    deltadt : datetime.timedelta object, optional
        Time resolution of the datetime series. The default is
        dt.timedelta(minutes=5).
    dtround : datetime.timedelta object, optional
        Round the dates to the specified component.
    toldt : datetime.timedelta object, optional
        Time window used to search within original date/time objects.
        The default is dt.timedelta(minutes=1).
    fillv : np.nan or bool, optional
        Value used to fill missing date/times. The default is np.nan.
    tz : str
        Key/name of the data timezone. The given tz string is then
        retrieved from the ZoneInfo module. The default is None.

    Returns
    -------
    dtsf : TYPE
        DESCRIPTION.
    valsn : TYPE
        DESCRIPTION.

    """
    # Creates the expected full time series
    if stspdt is None:
        dtsf = np.arange(dtso[0].replace(tzinfo=None),
                         dtso[-1].replace(tzinfo=None)+deltadt,
                         deltadt).astype(dt.datetime)
    else:
        # Set datetime min/max limits
        dtsf = np.arange(stspdt[0].replace(tzinfo=None),
                         stspdt[1].replace(tzinfo=None)+deltadt,
                         deltadt).astype(dt.datetime)
    # Adjust resolution of timeseries
    dtrs_nr = [time_round(suff, dtround).replace(tzinfo=None)
               for suff in dtso]
    dtrsf_nr = [time_round(suff, dtround).replace(tzinfo=None)
                for suff in dtsf]
    if tz is not None:
        dtrs_nr = [i.replace(tzinfo=ZoneInfo(tz)) for i in dtrs_nr]
        dtrsf_nr = [i.replace(tzinfo=ZoneInfo(tz)) for i in dtrsf_nr]

    # Convert datetime to float to make computation much faster
    dtrsts = np.array([i.timestamp() for i in dtrs_nr])
    dtrs_fullts = np.array([i.timestamp() for i in dtrsf_nr])
    if len(dtrs_fullts) != len(dtrsts):
        idxv = np.argwhere(np.isin(dtrs_fullts, dtrsts)).ravel()
        if idxv.size == dtrsts.size:
            # Some timesteps may create wrong indices, for instance datetimes
            # that changed beyond the dtround arg e.g.,
            # 1/1/2023T10:03 instead of 1/1/2023T10:05
            # These lines try to fix this issue. BUT, it takes much more time.
            if len(idxv) != len(dtrsts):
                # Previous code lines roll the ids beyond the right position.
                idxw = np.argmax(np.array([dtrs_fullts[i]-dtrsts[c]
                                           for c, i in enumerate(idxv)]) != 0)
                idxv = [idxv[c] if c < idxw
                        else rut.find_nearest(dtrs_fullts, i)
                        for c, i in enumerate(dtrsts)]
            if isinstance(valso, np.ndarray):
                valsn = np.empty_like(np.arange(dtrs_fullts.size))*fillv
            else:
                valsn = [fillv for i in dtrs_fullts]
            for c, i in enumerate(idxv):
                valsn[i] = valso[c]
        else:
            tsnear = [rut.find_nearest(dtrsts, idt) for idt in dtrs_fullts]
            tsrsdiff = [abs(dtrs_nr[ix]-dtrsf_nr[cnt])
                        for cnt, ix in enumerate(tsnear)]
            idxv = [tsnear[cnt] if tdif <= toldt else np.nan
                    for cnt, tdif in enumerate(tsrsdiff)]
            if isinstance(valso, np.ndarray):
                valsn = np.empty_like(np.arange(dtrs_fullts.size))*fillv
            else:
                valsn = [fillv for i in dtrs_fullts]
            for c, i in enumerate(idxv):
                if ~np.isnan(i):
                    valsn[c] = valso[i]
                else:
                    valsn[c] = fillv
            idxv = [tsnear[cnt] if tdif <= toldt else fillv
                    for cnt, tdif in enumerate(tsrsdiff)]
    else:
        warnings.warn('The date/time series have the same length!')
        valsn = valso
    if tz is not None:
        dtrsf_nr = [i.replace(tzinfo=ZoneInfo(tz)) for i in dtrsf_nr]
    if return_index:
        ret = (np.array(dtrsf_nr), valsn, idxv)
    else:
        ret = (np.array(dtrsf_nr), valsn)
    return ret


def mstats(estimates, target, rmse_norm='std'):
    """
    Compute metrics to assess estimate errors.

    Parameters
    ----------
    estimates : list or array
        Values to be evaluated.
    target : list or array
        Target values.
    rmse_norm : str
        Normalisation factor used for the computation of NRMSE. The string has
        to be one of 'std', 'rng', or 'mean'. The default is 'std'.

    Notes
    -----
    Scikit-learn is required.

    Returns
    -------
    dstats : dict
        N: Number of values \n
        R_Pearson [-]: Pearson correlation coefficients \n
        R [-]: Coefficient of determination R**2 \n
        MAE: Mean absolute error \n
        RMSE: Root Mean Squared Error \n
        NRMSE: Normalised Root Mean Squared Error \n
        NMB: Normalised Mean Bias \n
        KGE: Kling–Gupta efficiency \n
        KGEp: modified Kling–Gupta efficiency
    """
    import numpy as np
    import sklearn.metrics as skm
    estats = np.array([estimates, target]).T
    estats = estats[~np.isnan(estats).any(axis=1)]
    st_n = np.count_nonzero(estats[:, 0])
    # st_n = len(estats[:, 0])
    st_rp = np.corrcoef(estats[:, 1], estats[:, 0])
    st_r2 = skm.r2_score(estats[:, 1], estats[:, 0])
    st_mae = skm.mean_absolute_error(estats[:, 1], estats[:, 0])
    st_rmse = np.sqrt(skm.mean_squared_error(estats[:, 1], estats[:, 0]))
    st_nmb = np.nanmean(estats[:, 0] - estats[:, 1]) / np.nanmean(estats[:, 1])
    # st_nrmse = (np.sqrt(skm.mean_squared_error(estats[:, 1], estats[:, 0]))
    #             / (np.nanmean((estats[:, 1] - np.nanmean(estats[:, 1]))
    #                           ** 2)) ** 0.5)
    if rmse_norm == 'std':
        st_nrmse = (np.sqrt(skm.mean_squared_error(estats[:, 1], estats[:, 0]))
                    / np.nanstd(estats[:, 1], axis=-1, ddof=1))
    elif rmse_norm == 'mean':
        st_nrmse = (np.sqrt(skm.mean_squared_error(estats[:, 1], estats[:, 0]))
                    / np.nanmean(estats[:, 1]))
    elif rmse_norm == 'rng':
        st_nrmse = (np.sqrt(skm.mean_squared_error(estats[:, 1], estats[:, 0]))
                    / (np.nanmax(estats[:, 1]) - np.nanmin(estats[:, 1])))

    est_mean = np.nanmean(estimates, axis=0)
    tgt_mean = np.nanmean(target)
    # calculate r (Pearson's correlation coefficient)
    # r_num = np.nansum((estimates - est_mean) * (target - tgt_mean),
    #                   axis=0, dtype=np.float64)
    # r_den = np.sqrt(np.nansum((estimates - est_mean) ** 2,
    #                           axis=0, dtype=np.float64)
    #                 * np.nansum((target - tgt_mean) ** 2,
    #                             dtype=np.float64))
    # r = r_num / r_den
    r = st_rp[0][1]
    #  α is the relative variability between estimates and observed values
    alpha = np.nanstd(estimates, axis=0) / np.nanstd(target,
                                                     dtype=np.float64)
    # β is the bias ratio (dimensionless)
    # beta = (np.nansum(estimates, axis=0, dtype=np.float64)
    #         / np.nansum(target, dtype=np.float64))
    beta = (np.nanmean(estimates, axis=0, dtype=np.float64)
            / np.nanmean(target, dtype=np.float64))
    # γ is the variability ratio (dimensionless)
    gamma = ((np.nanstd(estimates, axis=0, dtype=np.float64) / est_mean)
             / (np.nanstd(estimates, dtype=np.float64) / tgt_mean))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    if kge_ < 0:
        kge_ = 0
    kge = {'kge': kge_, 'r': r, 'alpha': alpha, 'beta': beta}
    # calculate the modified Kling-Gupta Efficiency KGE
    kgeprime_ = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    kgep = {'kge': kgeprime_, 'r': r, 'alpha': alpha, 'beta': beta,
            'gamma': gamma}

    dstats = {'N': st_n, 'R_Pearson [-]': st_rp, 'R [-]': st_r2,
              'MAE': st_mae, 'RMSE': st_rmse, 'NRMSE [%]': st_nrmse*100,
              'NMB [%]': st_nmb*100, 'KGE': kge, 'KGEp': kgep}

    return dstats


# =============================================================================
# io
# =============================================================================
def get_listfilesxpol(radar_site, start_time, stop_time, scan_elev=None,
                      parent_dir=None, working_dir=None):
    """
    Yield a list of radar files for a given period of time and location.

    Parameters
    ----------
    radar_site : str
        Name of the radar site. The string has to be one of 'boxpol'
        or ‘juxpol.
    start_time : datetime
        Datetime object defining the initial date and time to include when
        searching for files.
    stop_time : datetime
        Datetime object defining the final date and time to include when
        searching for files.
    scan_elev : str
        Choose accordingly depending on the radar site:
        Boxpol: ['n_vertical_scan', 'n_ppi_280deg', 'n_ppi_180deg',
                 'n_ppi_140deg', 'n_ppi_110deg', 'n_ppi_082deg',
                 'n_ppi_060deg', 'n_ppi_045deg', 'n_ppi_031deg',
                 'n_ppi_020deg', 'n_ppi_010deg']
        Juxpol: [None, 'Vert']
    parent_dir : str, optional
       Modifies the parent directory in the local server where the radar data
       are stored. The default is None.
    working_dir : str, optional
       Ignores all args and reads all files within the specified directory.
       The default is None.

    Returns
    -------
    list of radar files : list

    """
    if parent_dir is None:
        wdir1 = '/automount/'
    else:
        wdir1 = parent_dir
    if stop_time < dt.datetime(2015, 1, 1):
        oldf = "-archiv"
    else:
        oldf = ""
    if radar_site.lower() == 'boxpol':
        if working_dir is not None:
            wdir = working_dir
        else:
            if (dt.datetime(start_time.year, start_time.month, start_time.day)
                == dt.datetime(stop_time.year, stop_time.month,
                               stop_time.day)):
                wdir = (f'{wdir1}radar{oldf}/scans/{start_time.year}/'
                        f'{start_time.year}-'+f'{start_time.month}'.zfill(2)
                        + '/'
                        + f'{start_time.year}-'+f'{start_time.month}'.zfill(2)
                        + '-' + f'{start_time.day}'.zfill(2)
                        + f'/{scan_elev}/')
                # wdir[:] = [i for i in wdir if os.path.isdir(i)]
                # nfilesdt = [n for n in sorted(os.listdir(wdir))]
                if os.path.isdir(wdir):
                    nfilesdt = [n for n in sorted(os.listdir(wdir))]
                else:
                    nfilesdt = []
            else:
                dtd = [i for i in np.arange((stop_time
                                             - start_time).total_seconds()
                                            / 86400 + 1)]
                wdt = [start_time+dt.timedelta(days=i) for i in dtd]
                wdir = [(f'{wdir1}radar{oldf}/scans/{i.year}/' f'{i.year}-'
                         + f'{i.month}'.zfill(2) + '/' + f'{i.year}-'
                         + f'{i.month}'.zfill(2) + '-' + f'{i.day}'.zfill(2)
                         + f'/{scan_elev}/') for i in wdt]
                wdir[:] = [i for i in wdir if os.path.isdir(i)]
                nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                nfilesdt[:] = [j for i in nfilesdt for j in i]
        if stop_time < dt.datetime(2017, 4, 16, 0, 0):
            nfilesdt[:] = [n.removesuffix(',00.mvol') for n in nfilesdt]
            nfilesdt[:] = [dt.datetime.strptime(n, "%Y-%m-%d--%H:%M:%S")
                           for n in nfilesdt]
        else:
            nfilesdt[:] = [n.removeprefix(f'{scan_elev}_12345_')
                           for n in nfilesdt]
            nfilesdt[:] = [n.removesuffix('_00.h5') for n in nfilesdt]
            nfilesdt[:] = [dt.datetime.strptime(n, "%Y%m%d%H%M%S")
                           for n in nfilesdt]
    elif radar_site.lower() == 'juxpol':
        if working_dir is not None:
            wdir = working_dir
        else:
            if (dt.datetime(start_time.year, start_time.month, start_time.day)
                == dt.datetime(stop_time.year, stop_time.month,
                               stop_time.day)):
                # try:
                if scan_elev == 'Vert':
                    wdir = (
                        f'{wdir1}radar{oldf}/scans_juelich/{start_time.year}/'
                        f'{start_time.year}-' + f'{start_time.month}'.zfill(2)
                        + '/' + f'{start_time.year}-'
                        + f'{start_time.month}'.zfill(2) + '-'
                        + f'{start_time.day}'.zfill(2) + f'/{scan_elev}/')
                    nfilesdt = [n for n in sorted(os.listdir(wdir))]
                # except FileNotFoundError:
                else:
                    wdir = (
                        f'{wdir1}radar{oldf}/scans_juelich/{start_time.year}/'
                        f'{start_time.year}-' + f'{start_time.month}'.zfill(2)
                        + '/' + f'{start_time.year}-'
                        + f'{start_time.month}'.zfill(2) + '-'
                        + f'{start_time.day}'.zfill(2) + '/DWD_Vol_2/')
                    nfilesdt = [n for n in sorted(os.listdir(wdir))]
            else:
                dtd = [i for i in np.arange((stop_time
                                             - start_time).total_seconds()
                                            / 86400 + 1)]
                wdt = [start_time+dt.timedelta(days=i) for i in dtd]
                try:
                    wdir = [(f'{wdir1}radar{oldf}/scans_juelich/{i.year}/'
                             + f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '/' f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '-'
                             + f'{i.day}'.zfill(2)
                             + f'/{scan_elev}/') for i in wdt]
                    nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                    nfilesdt[:] = [j for i in nfilesdt for j in i]
                    # break
                except FileNotFoundError:
                    wdir = [(f'{wdir1}radar{oldf}/scans_juelich/{i.year}/'
                             + f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '/' f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '-'
                             + f'{i.day}'.zfill(2)
                             + '/DWD_Vol_2/') for i in wdt]
                    wdir[:] = [i for i in wdir if os.path.isdir(i)]
                    nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                    nfilesdt[:] = [j for i in nfilesdt for j in i]
        # if start_time.year < 2017:
        if stop_time < dt.datetime(2017, 10, 1, 0, 0):
            nfilesdt[:] = [n.removesuffix(',00.mvol') for n in nfilesdt]
            nfilesdt[:] = [dt.datetime.strptime(n, "%Y-%m-%d--%H:%M:%S%f")
                           for n in nfilesdt]
        else:
            nfilesdt[:] = [n.removeprefix('Vert_99999_')
                           if n.startswith('Vert_99999_') else n
                           for n in nfilesdt]
            nfilesdt[:] = [n.removeprefix('DWD-Vol-2_99999_')
                           if n.startswith('DWD-Vol-2_99999_') else n
                           for n in nfilesdt]

            nfilesdt[:] = [n.removesuffix('_00.h5') for n in nfilesdt]
            nfilesdt[:] = [dt.datetime.strptime(n, "%Y%m%d%H%M%S")
                           for n in nfilesdt]

    elif radar_site.lower() == 'aaxpol':
        if working_dir is not None:
            wdir = working_dir
        else:
            if (dt.datetime(start_time.year, start_time.month, start_time.day)
                == dt.datetime(stop_time.year, stop_time.month,
                               stop_time.day)):
                try:
                    wdir = (f'{wdir1}realpep/upload/aachen-data/'
                            + f'{start_time.year}/{start_time.year}-'
                            + f'{start_time.month}'.zfill(2)
                            + '/' + f'{start_time.year}-'
                            + f'{start_time.month}'.zfill(2) + '-'
                            + f'{start_time.day}'.zfill(2) + '/')
                    # wdir[:] = [i for i in wdir if os.path.isdir(i)]
                    nfilesdt = [n for n in sorted(os.listdir(wdir))]
                    # break
                except FileNotFoundError:
                    wdir = (f'{wdir1}realpep/upload/aachen-data/'
                            + '{start_time.year}/'
                            f'{start_time.year}-'
                            + f'{start_time.month}'.zfill(2)
                            + '/' + f'{start_time.year}-'
                            + f'{start_time.month}'.zfill(2) + '-'
                            + f'{start_time.day}'.zfill(2) + '/')
                    # wdir[:] = [i for i in wdir if os.path.isdir(i)]
                    nfilesdt = [n for n in sorted(os.listdir(wdir))]
                # raise TowerpyError('No such directory')
            else:
                dtd = [i for i in np.arange((stop_time
                                             - start_time).total_seconds()
                                            / 86400 + 1)]
                wdt = [start_time+dt.timedelta(days=i) for i in dtd]
                try:
                    wdir = [(f'{wdir1}realpep/upload/aachen-data/{i.year}/'
                             + f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '/' f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '-'
                             + f'{i.day}'.zfill(2) + '/') for i in wdt]
                    nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                    nfilesdt[:] = [j for i in nfilesdt for j in i]
                    # break
                except FileNotFoundError:
                    wdir = [(f'{wdir1}realpep/upload/aachen-data/{i.year}/'
                             + f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '/' f'{i.year}-'
                             + f'{i.month}'.zfill(2) + '-'
                             + f'{i.day}'.zfill(2) + '/') for i in wdt]
                    wdir[:] = [i for i in wdir if os.path.isdir(i)]
                    nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                    nfilesdt[:] = [j for i in nfilesdt for j in i]
        if start_time.year >= 2023:
            nfilesdt[:] = [n.removeprefix('vol_')
                           if n.startswith('vol_') else n
                           for n in nfilesdt]
            nfilesdt[:] = [n.removesuffix('.h5') for n in nfilesdt]
            nfilesdt[:] = [dt.datetime.strptime(n, "%Y%m%d%H%M")
                           for n in nfilesdt]
    if nfilesdt:
        nfilesidx = [i for i, d in enumerate(nfilesdt)
                     if d >= start_time and d <= stop_time]

        if (dt.datetime(start_time.year, start_time.month, start_time.day)
            == dt.datetime(stop_time.year, stop_time.month,
                           stop_time.day)):
            nfilesf = [wdir+f for n, f in enumerate(sorted(os.listdir(wdir)))
                       if n in nfilesidx]
        else:
            nfilesf = [sorted(os.listdir(n)) for n in wdir]
            nfilesf[:] = [z+j for i, z in zip(nfilesf, wdir) for j in i]
            nfilesf[:] = [v for i, v in enumerate(nfilesf) if i in nfilesidx]
        # listof_radarfiles = {'files_path': nfiles, 'files_dt': nfilesdt2}
        listof_radarfiles = nfilesf
    else:
        warnings.warn('There are no files related to the chosen elevation or'
                      ' date/time. Check the inputs!')
        listof_radarfiles = []
    return listof_radarfiles


def get_listfilesdwd(radar_site, start_time, stop_time, scan_elev='ppi_pcp',
                     parent_dir=None, working_dir=None):
    """
    Yield a list of radar files for a given period of time and location.

    Parameters
    ----------
    radar_site : str
        Name of the radar site part of the DWD radar network:
            ['ASR Borkum', 'Boostedt', 'Dresden', 'Eisberg', 'Essen',
            'Feldberg', 'Flechtdorf', 'Hannover', 'Isen', 'Memmingen',
            'Neuhaus', 'Neuheilenbach', 'Offenthal', 'Protzel', 'Rostock',
            'Turkheim', 'Ummendorf']
    start_time : datetime
        Datetime object defining the initial date and time to include when
        searching for files.
    stop_time : datetime
        Datetime object defining the final date and time to include when
        searching for files.
    scan_elev : str
        Scan elevation according to the DWD scan strategy:
            ['ppi_pcp', 'ppi_vol_0.5', 'ppi_vol_1.5', 'ppi_vol_2.5',
            'ppi_vol_3.5', 'ppi_vol_4.5', 'ppi_vol_5.5', 'ppi_vol_8.0',
            'ppi_vol_12.0', 'ppi_vol_17.0', 'ppi_vol_25.0', 'ppi_vrt_90g']
        The default is 'ppi_pcp'.
    parent_dir : str, optional
       Modifies the parent directory in the local server where the radar data
       are stored. The default is None.
    working_dir : str, optional
       Ignores all args and reads all files within the specified directory.
       The default is None.

    Notes
    -----
    1. Radar site names according to [1]_
    2. Scanning strategy as described in [2]_

    References
    ----------
    .. [1] Deutscher Wetterdienst. (2018). Metadaten zu den Radaren des
        Radarverbunds des DWD.
        Retrieved from (https://www.dwd.de/DE/derdwd/messnetz/\
atmosphaerenbeobachtung/_functions/HaeufigGesucht/koordinaten-radarverbund.pdf\
?__blob=publicationFile&v=5)
    .. [2] DWD. “Radar products.”
    https://www.dwd.de/EN/ourservices/radar_products/radar_products.html

    Returns
    -------
    list of radar files : list

    """
    if parent_dir is None:
        wdir1 = '/automount/'
    else:
        wdir1 = parent_dir
    dwd_sites = {'ASR Borkum': 'asb', 'Boostedt': 'boo', 'Dresden': 'drs',
                 'Eisberg': 'eis', 'Essen': 'ess', 'Feldberg': 'fbg',
                 'Flechtdorf': 'fld', 'Hannover': 'hnr', 'Isen': 'isn',
                 'Memmingen': 'mem', 'Neuhaus': 'neu', 'Neuheilenbach': 'nhb',
                 'Offenthal': 'oft', 'Protzel': 'pro', 'Rostock': 'ros',
                 'Turkheim': 'tur', 'Ummendorf': 'umd'}
    selev_dic = {'ppi_vol_5.5': '00', 'ppi_vol_4.5': '01',
                 'ppi_vol_3.5': '02', 'ppi_vol_2.5': '03',
                 'ppi_vol_1.5': '04', 'ppi_vol_0.5': '05',
                 'ppi_vol_8.0': '06', 'ppi_vol_12.0': '07',
                 'ppi_vol_17.0': '08', 'ppi_vol_25.0': '09',
                 'ppi_vrt_90g': '00', 'ppi_pcp': '00'}
    rdsite = dwd_sites.get(radar_site)
    # if stop_time < dt.datetime(2022, 12, 21):
    stype_dic = {'pcp': 'pcpng01', 'vol': 'vol5minng01',
                 '90g': '90gradstarng01'}
    # else:
    #     stype_dic = {'pcp': 'pcpng10', 'vol': 'vol5minng10',
    #                  '90g': '90gradstarng10'}
    if '_vol_' in scan_elev:
        scan_type = 'vol'
    elif '_pcp' in scan_elev:
        scan_type = 'pcp'
    elif '_vrt_' in scan_elev:
        scan_type = '90g'
    stype = stype_dic.get(scan_type)
    selev = selev_dic.get(scan_elev, '00')

    if working_dir is not None:
        wdir = working_dir
        nfilesdt = [n for n in sorted(os.listdir(wdir)) if rdsite in n]
        nfilesdt[:] = [n for n in sorted(nfilesdt)
                       if start_time.strftime('%Y%m%d') in n]
        nfilesdt2 = copy.deepcopy(nfilesdt)
    else:
        if (dt.datetime(start_time.year, start_time.month, start_time.day)
            == dt.datetime(stop_time.year, stop_time.month,
                           stop_time.day)):
            wdir = (f'{wdir1}realpep/upload/RealPEP-SPP/DWD-CBand/'
                    f'{start_time.year}/' + f'{start_time.year}-'
                    + f'{start_time.month}'.zfill(2) + '/'
                    + f'{start_time.year}-'
                    + f'{start_time.month}'.zfill(2) + '-'
                    + f'{start_time.day}'.zfill(2) + '/'
                    + rdsite + '/' + f'{stype}'
                    + '/' + f'{selev}' + '/')
            if os.path.isdir(wdir):
                nfilesdt = [n for n in sorted(os.listdir(wdir))]
            else:
                nfilesdt = []
        else:
            dtd = [i for i in np.arange((stop_time
                                         - start_time).total_seconds()
                                        / 86400 + 1)]
            wdt = [start_time+dt.timedelta(days=i) for i in dtd]
            wdir = [(f'{wdir1}realpep/upload/RealPEP-SPP/DWD-CBand/'
                    f'{i.year}/' + f'{i.year}-' + f'{i.month}'.zfill(2)
                     + '/' f'{i.year}-' + f'{i.month}'.zfill(2) + '-'
                     + f'{i.day}'.zfill(2) + '/'
                     + rdsite + '/' + f'{stype}'
                     + '/' + f'{selev}' + '/') for i in wdt]
            try:
                nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                nfilesdt[:] = [j for i in nfilesdt for j in i]
            except FileNotFoundError:
                wdir[:] = [i for i in wdir if os.path.isdir(i)]
                nfilesdt = [sorted(os.listdir(n)) for n in wdir]
                nfilesdt[:] = [j for i in nfilesdt for j in i]
    if nfilesdt:
        # if stop_time < dt.datetime(2022, 12, 21):
        kw_idf = 'sweeph5onem_'  # prefix identifier within the file name
        prfx = 'ras07'
        if scan_type == '90g':
            kw_idf = 'sweeph5allm_'  # prefix identifier
            prfx = 'ras11'
            stype = '90gradstarng10'
        # else:
        #     kw_idf = 'sweeph5allm_any'  # prefix identifier
        #     prfx = 'ras11'
        #     if scan_type == '90g':
        #         kw_idf = 'sweeph5allm_'  # prefix identifier
        #         prfx = 'ras11'
        #         stype = '90gradstarng10'
        nfilesdt[:] = [n.removeprefix(f'{prfx}-{stype}_{kw_idf}')
                       for n in nfilesdt]
        nfilesdt[:] = [n[n.find('-')+1:] for n in nfilesdt]
        nfilesdt[:] = [n.removesuffix('hd5') for n in nfilesdt]
        nfilesdt[:] = [n[:n.find('-')] for n in nfilesdt]
        # Make sure to only get pvars for ras11
        # if stop_time > dt.datetime(2022, 12, 21):
        #     dmmydt = dt.datetime(1900, 1, 1).strftime("%Y%m%d%H%M%S")
        #     nfilesdt[:] = [n if n.isnumeric() else dmmydt for n in nfilesdt]
        nfilesdt[:] = [dt.datetime.strptime(n, '%Y%m%d%H%M%S%f')
                       for n in nfilesdt]
        nfilesidx = [i for i, d in enumerate(nfilesdt)
                     if d >= start_time and d <= stop_time]
        if not nfilesidx:
            warnings.warn('There are no files related to the chosen elevation'
                          ' or date/time. Check the inputs!')
        nfilesdtv = [f for n, f in enumerate(nfilesdt) if n in nfilesidx]
        nfilesdtu = sorted(list(set(nfilesdtv)))
        if (dt.datetime(start_time.year, start_time.month, start_time.day)
            == dt.datetime(stop_time.year, stop_time.month,
                           stop_time.day)):
            nfilesf = [wdir+f for n, f in enumerate(sorted(os.listdir(wdir)))
                       if n in nfilesidx]
        else:
            if working_dir is not None:
                nfilesf = nfilesdt2
                # nfilesf =  deepcopy(nfilesdt2)
                nfilesf[:] = [wdir+nf for nf in nfilesf]
            else:
                nfilesf = [sorted(os.listdir(n)) for n in wdir]
                nfilesf[:] = [z+j for i, z in zip(nfilesf, wdir) for j in i]
            nfilesf[:] = [v for i, v in enumerate(nfilesf) if i in nfilesidx]
        # Group by timestamp
        if len(nfilesdtu) > 1 and prfx != 'ras11':
            nfilesdtg = {i: [] for i in nfilesdtu}
            for i, v in enumerate(nfilesdtv):
                nfilesdtg[v].append(i)
            # listof_radarfiles = {k: [nfilesf[i] for i in v]
            #                      for k, v in nfilesdtg.items()}
            listof_radarfiles = [[nfilesf[i] for i in v]
                                 for k, v in nfilesdtg.items()]
        else:
            listof_radarfiles = nfilesf
    else:
        warnings.warn('There are no files related to the chosen elevation or'
                      ' date/time. Check the inputs!')
        listof_radarfiles = []
    return listof_radarfiles


def netcdf2pydict(ncfile, vars2read=None):
    """Read netCDF data into a dictionary.

    Parameters
    ----------
    ncfile : string
        Location and name of the netCDF file to be read
    vars2read : string list, optional
        List of variables to be read from file. If None, all the variables are
        retrieved. The default is None
    Returns
    -------
    ncdict : A python dict containing the files variables

    Examples
    --------
    >>> d=netcdf2pydict('netcdffile.nc')
    >>> d2=netcdf2pydict('netcdffile.nc', vars2read=['var1', 'var2'])
    """
    try:
        ncdata = nc.Dataset(ncfile)
    except IOError:
        warnings.warn('Error: Cannot open file' + ncfile)

    if vars2read is None:
        vars2read = ncdata.variables.keys()
        ncdict = dict((f'{v} [{ncdata.variables[v].unit}]',
                       ncdata.variables[v][:].data)
                      if hasattr(ncdata.variables[v], "unit")
                      else (v, ncdata.variables[v][:])
                      for v in vars2read)
        times = ncdata.variables['time']
        dt_nc = nc.num2date(times[:], units=times.units)
        ncdict['time'] = [dt.datetime.strptime(i.isoformat(timespec='seconds'),
                                               '%Y-%m-%dT%H:%M:%S')
                          for i in dt_nc]
    else:
        ncdict = dict((f'{v} [{ncdata.variables[v].unit}]',
                       ncdata.variables[v][:].data)
                      if hasattr(ncdata.variables[v], "unit")
                      else (v, ncdata.variables[v][:])
                      for v in vars2read)
        times = ncdata.variables['time']
        dt_nc = nc.num2date(times[:], units=times.units)
        ncdict['time'] = [dt.datetime.strptime(i.isoformat(),
                                               '%Y-%m-%dT%H:%M:%S')
                          for i in dt_nc]
    return ncdict


class Rad_scan:
    """A Towerpy class to store radar scans data."""

    def __init__(self, rfile_path, radar_site):
        """
        Init a Rad_scan Class.

        Parameters
        ----------
        rfile_path : string
            Path file containing the radar data.
        radar_site : string
            Name of the radar site.
        """
        self.file_name = rfile_path
        self.site_name = radar_site

    def ppi_xpol(self, get_rvar='pvars', get_rawvars=True, scan_elev='sweep_0',
                 rcalc_zdr=False, tz='Europe/Berlin'):
        """
        Read-in UniBonn X-band PPI radar scans using wradlib.

        Parameters
        ----------
        get_rvar : list or string, optional
            Specifies the radar variables to read as a string or as a list
            of strings specifying the variables within the file to read.
            The string has to be one of ‘pvars’, ‘all’, or list using the
            identifiers specified in [1]_. The default is 'pvars'.
        get_rawvars : bool, optional
            Specifies if the radar variables are retrieved as
            filtered/processed (False) or uncorrected/raw (True).
            The default is True.
        scan_elev : str, optional
            Descriptor for the sweep related to the scan elevation, applicable
            only for JuXPol scan files. See [2]_ for more details.
            The default is 'sweep_0'.
        rcalc_zdr : bool, optional
            Compute ZDR from ZH and ZV. The default is False.
        tz : str, optional
            Key/name of the radar data timezone. The given tz value is then
            retrieved from the ZoneInfo module. The default is 'Europe/Berlin'.

        Returns
        -------
        Towerpy radar object.

        Notes
        -----
        .. [1] List of variables in the XPol radars:

                'DBZH', 'DBZV', 'UPHIDP', 'KDP', 'RHOHV', 'URHOHV', 'CMAP',
                'DBTH', 'DBTV', 'ZDR', 'VRADH', 'VRADV', 'WRADH', 'WRADV',
                'PHIDP'
        .. [2] Sweep number for each scan elevation:

            .. glossary::

                'el_280': 'sweep_0'
                'el_180': 'sweep_1'
                'el_140': 'sweep_2'
                'el_110': 'sweep_3'
                'el_82': 'sweep_4'
                'el_60': 'sweep_5'
                'el_45': 'sweep_6'
                'el_31': 'sweep_7'
                'el_17': 'sweep_8'
                'el_06': 'sweep_9'
                'Vert': 'sweep_0'
        """
        xpol_vars = {'DBZH': {'raw': False, 'default': True,
                              'tpkey': 'ZH [dBZ]'},
                     'DBZV': {'raw': False, 'default': False,
                              'tpkey': 'ZV [dBZ]'},
                     'DBTH': {'raw': True, 'default': True,
                              'tpkey': 'ZH [dBZ]'},
                     'DBTV': {'raw': True, 'default': False,
                              'tpkey': 'ZV [dBZ]'},
                     'ZDR': {'raw': 'Proc', 'default': True,
                             'tpkey': 'ZDR [dB]'},
                     'PHIDP': {'raw': False, 'default': True,
                               'tpkey': 'PhiDP [deg]'},
                     'UPHIDP': {'raw': True, 'default': True,
                                'tpkey': 'PhiDP [deg]'},
                     'RHOHV': {'raw': False, 'default': True,
                               'tpkey': 'rhoHV [-]'},
                     'URHOHV': {'raw': True, 'default': True,
                                'tpkey': 'rhoHV [-]'},
                     'VRADH': {'raw': 'Proc', 'default': True,
                               'tpkey': 'V [m/s]'},
                     'VRADV': {'raw': 'Proc', 'default': False,
                               'tpkey': 'VV [m/s]'},
                     'KDP': {'raw': 'Proc', 'default': False,
                             'tpkey': 'KDP [deg/km]'},
                     'WRADH': {'raw': 'Proc', 'default': False,
                               'tpkey': 'W [m/s]'},
                     'WRADV': {'raw': 'Proc', 'default': False,
                               'tpkey': 'WV [m/s]'},
                     }
        if get_rvar == 'all':
            radvars = {k: v for k, v in xpol_vars.items()}
        elif get_rvar == 'pvars':
            radvars = {k: v for k, v in xpol_vars.items()
                       if v['default']}
            if get_rawvars:
                radvars = {k: v['tpkey'] for k, v in radvars.items()
                           if v['raw'] is not False}
            else:
                radvars = {k: v['tpkey'] for k, v in radvars.items()
                           if v['raw'] is not True}
        else:
            radvars = {k: v['tpkey'] for k, v in xpol_vars.items()
                       if k in get_rvar}

        # =====================================================================
        # Fill up TP object using WR data
        # =====================================================================
        wrobj = xr.open_dataset(self.file_name, engine='gamic',
                                group=scan_elev,
                                reindex_angle=dict(start_angle=0,
                                                   stop_angle=360,
                                                   angle_res=1.0, direction=1))
        wrobj.wrl.georef.georeference(crs=wrl.georef.get_earth_projection())
        if get_rvar == 'pvars' or get_rvar == 'all':
            if (get_rawvars and 'URHOHV' not in wrobj.keys()
               and 'RHOHV' in wrobj.keys()):
                radvars['RHOHV'] = radvars.pop('URHOHV')
                warnings.warn('URHOHV not in data file')
            if (get_rawvars and 'UPHIDP' not in wrobj.keys()
               and 'PHIDP' in wrobj.keys()):
                radvars['PHIDP'] = radvars.pop('UPHIDP')
                warnings.warn('UPHIDP not in data file')
        else:
            avars = {k: k in list(wrobj.keys()) for k in get_rvar}
            if not all(k in list(wrobj.keys()) for k in get_rvar):
                raise NameError(
                    'Input variable(s) '
                    + f' {*[k for k, v in avars.items() if v is False],}'
                    + ' cannot be found in the radar file \n'
                    + f'Available variables: {*list(wrobj.keys()),}')
        warnings.warn(
            'The following default variables'
            + f' {*[k for k in radvars.keys() if k not in wrobj.keys()],}'
            + 'are not in the file.')
        radvars = {k: v for k, v in radvars.items() if k in wrobj.keys()}

        rvars = {}
        for k, v in radvars.items():
            rvars[v] = wrobj[k].values.astype(np.float64)
        if rcalc_zdr and get_rawvars:
            zhr = wrobj.DBTH.values.astype(np.float64)
            zvr = wrobj.DBTV.values.astype(np.float64)
            zhrl = 10**(np.array(zhr)/10)
            zvrl = 10**(np.array(zvr)/10)
            rvars['ZDR [dB]'] = 10*np.log10(zhrl/zvrl)

        rparams = {'nvars': len(radvars),
                   'ngates': wrobj.range.shape[0],
                   'nrays': wrobj.azimuth.shape[0],
                   'pulselength [ns]': np.nan,
                   'avsamples': np.nan}
        rparams['site_name'] = self.site_name
        ns = 1e-9  # number of seconds in a nanosecond
        scandt = dt.datetime.utcfromtimestamp(wrobj.time.values[0].astype(int)
                                              * ns)
        rparams['datetime'] = scandt.replace(tzinfo=ZoneInfo(tz))
        dtarrayx = list(rparams['datetime'].timetuple())[: -3]
        rparams['datetimearray'] = dtarrayx
        rparams['altitude [m]'] = float(wrobj.altitude.values)
        rparams['rpm'] = np.nan,
        rparams['latitude [dd]'] = float(wrobj.latitude.values)
        rparams['longitude [dd]'] = float(wrobj.longitude.values)
        rparams['elev_ang [deg]'] = wrobj.sweep_fixed_angle.values
        rparams['range_start [m]'] = wrobj.bins.values[0, 0]
        dm1 = wrobj.bins.values[0, 1] - wrobj.bins.values[0, 0]
        rparams['gateres [m]'] = dm1
        if self.site_name.lower() == 'boxpol':
            rparams['beamwidth [deg]'] = 1.
            rparams['frequency [GHz]'] = 9.3
            wv = (sc.c / rparams['frequency [GHz]'])/10000000
            rparams['wavelength [cm]'] = wv
            if rparams['elev_ang [deg]'] == 1.:
                rparams['prf [Hz]'] = 700
            elif rparams['elev_ang [deg]'] == 2.:
                rparams['prf [Hz]'] = 800
            elif rparams['elev_ang [deg]'] == 3.1:
                rparams['prf [Hz]'] = 900
            elif rparams['elev_ang [deg]'] == 4.5:
                rparams['prf [Hz]'] = 950
            elif rparams['elev_ang [deg]'] == 6.:
                rparams['prf [Hz]'] = 1050
            else:
                rparams['prf [Hz]'] = 1150
        else:
            rparams['beamwidth [deg]'] = 1.
            rparams['frequency [GHz]'] = 9.3
            wv = (sc.c / rparams['frequency [GHz]'])/10000000
            rparams['wavelength [cm]'] = wv
        rparams['radar constant [dB]'] = 0.
        rparams['site_name'] = self.site_name.lower()

        # =====================================================================
        # Create a georeference grid
        # =====================================================================
        elev = np.deg2rad(wrobj.elevation.values)
        azim = np.deg2rad(wrobj.azimuth.values)
        rng = wrobj.range.values.astype(np.float64)
        rh, th = np.meshgrid(rng/1000, azim)

        bhkm = np.array([geo.height_beamc(ray, rng/1000)
                         for ray in np.rad2deg(elev)])

        bbhkm = np.array([geo.height_beamc(ray-rparams['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])

        bthkm = np.array([geo.height_beamc(ray+rparams['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])

        s = np.array([geo.cartesian_distance(ray, rng/1000, bhkm[0])
                      for i, ray in enumerate(np.rad2deg(elev))])
        a = [geo.pol2cart(arcl, azim) for arcl in s.T]
        xgrid = np.array([i[1] for i in a]).T
        ygrid = np.array([i[0] for i in a]).T

        geogrid = {'azim [rad]': azim,
                   'elev [rad]': elev,
                   'range [m]': rng,
                   'rho': rh,
                   'theta': th,
                   'grid_rectx': xgrid,
                   'grid_recty': ygrid}

        geogrid['beam_height [km]'] = bhkm
        geogrid['beambottom_height [km]'] = bbhkm
        geogrid['beamtop_height [km]'] = bthkm
        igetp = list(wrobj.keys())[0]

        geogrid['grid_wgs84x'] = wrobj[igetp].x.values
        geogrid['grid_wgs84y'] = wrobj[igetp].y.values

        # wrobj.wrl.georef.reproject(trg_crs=wrl.georef.epsg_to_osr(32632))
        # wrobj.wrl.georef.reproject(trg_crs=wrl.georef.epsg_to_osr(25832))
        proj_utm = wrl.georef.epsg_to_osr(32632)
        xpol_coord = wrl.georef.spherical_to_centroids(wrobj, crs=proj_utm)
        geogrid['grid_utmx'] = xpol_coord[:, :, 0].values
        geogrid['grid_utmy'] = xpol_coord[:, :, 1].values

        self.georef = geogrid
        self.params = rparams
        self.vars = rvars
        self.elev_angle = rparams['elev_ang [deg]']
        self.scandatetime = rparams['datetime']

    def ppi_dwd(self, get_rvar='pvars', get_rawvars=True, tz='Europe/Berlin'):
        """
        Read radar data from DWD PPI binary files (hd5) using wradlib.

        Parameters
        ----------
        get_rvar : list or string, optional
            Specifies the radar variables to read as a string or as a list
            of strings specifying the variables within the file to read.
            The string has to be one of ‘pvars’, ‘all’, or list using the
            identifiers specified in [1]_. The default is 'pvars'.
        get_rawvars : bool, optional
            Specifies if the radar variables are retrieved as
            filtered/processed (False) or uncorrected/raw (True).
            The default is True.
        tz : str, optional
            Key/name of the radar data timezone. The given tz value is then
            retrieved from the ZoneInfo module. The default is 'Europe/Berlin'.

        Notes
        -----
        1. Pass a list to the file_name attribute only for volume or
        precipitation scans. The files within the list must share the
        scan's timestamp exactly. When reading vertical scans, make sure to
        pass a string referring to a single scan only.

        References
        ----------
        .. [1] EUMETNET OPERA 4. (2019). EUMETNET OPERA weather radar
        information model for implementation with the HDF5 file format,
        Version 2.3. Retrieved from
        https://eumetnet.eu/wp-content/uploads/2019/01/ODIM_H5_v23.pdf

        Returns
        -------
        Towerpy radar object.

        """
        orvarname = {'dbzh': {'raw': False, 'default': True,
                              'tpkey': 'ZH [dBZ]'},
                     'dbzv': {'raw': False, 'default': False,
                              'tpkey': 'ZV [dBZ]'},
                     'th': {'raw': True, 'default': True,
                            'tpkey': 'ZH [dBZ]'},
                     'tv': {'raw': True, 'default': False,
                            'tpkey': 'ZV [dBZ]'},
                     'uzdr': {'raw': True, 'default': True,
                              'tpkey': 'ZDR [dB]'},
                     'zdr1': {'raw': 'Proc', 'default': False,
                              'tpkey': 'ZDRb [dB]'},
                     'zdr': {'raw': False, 'default': True,
                             'tpkey': 'ZDR [dB]'},
                     'uzdr1': {'raw': 'Proc', 'default': False,
                               'tpkey': 'ZDRb [dB]'},
                     'uphidp': {'raw': True, 'default': True,
                                'tpkey': 'PhiDP [deg]'},
                     'rhohv': {'raw': False, 'default': True,
                               'tpkey': 'rhoHV [-]'},
                     'urhohv': {'raw': True, 'default': True,
                                'tpkey': 'rhoHV [-]'},
                     'fvradh': {'raw': 'Proc', 'default': False,
                                'tpkey': 'V [m/s]'},
                     'vradh': {'raw': False, 'default': True,
                               'tpkey': 'V [m/s]'},
                     'vradv': {'raw': False, 'default': False,
                               'tpkey': 'VV [m/s]'},
                     'uvradh': {'raw': True, 'default': True,
                                'tpkey': 'V [m/s]'},
                     'uvradv': {'raw': True, 'default': False,
                                'tpkey': 'VV [m/s]'},
                     'ccorh': {'raw': False, 'default': False,
                               'tpkey': 'ccorh [dB]'},
                     'ccorv': {'raw': False, 'default': False,
                               'tpkey': 'ccorv [dB]'},
                     'cflags': {'raw': False, 'default': False,
                                'tpkey': 'cflags [class]'},
                     'cmap': {'raw': False, 'default': False,
                              'tpkey': 'cmap [0-1]'},
                     'cpah': {'raw': False, 'default': False,
                              'tpkey': 'cpah [0-1]'},
                     'cpav': {'raw': False, 'default': False,
                              'tpkey': 'cpav [0-1]'},
                     'snrhc': {'raw': False, 'default': False,
                               'tpkey': 'snrhc [dB]'},
                     'snrvc': {'raw': False, 'default': False,
                               'tpkey': 'snrvc [dB]'},
                     'sqi2h': {'raw': False, 'default': False,
                               'tpkey': 'sqi2h [0-1]'},
                     'sqi2v': {'raw': False, 'default': False,
                               'tpkey': 'sqi2v [0-1]'},
                     'sqi3h': {'raw': False, 'default': False,
                               'tpkey': 'sqi3h [0-1]'},
                     'sqi3v': {'raw': False, 'default': False,
                               'tpkey': 'sqi3v [0-1]'},
                     'sqih': {'raw': False, 'default': False,
                              'tpkey': 'sqih [0-1]'},
                     'sqiv': {'raw': False, 'default': False,
                              'tpkey': 'sqiv [0-1]'},
                     'stdh': {'raw': False, 'default': False,
                              'tpkey': 'stdh [unk]'},
                     'stdv': {'raw': False, 'default': False,
                              'tpkey': 'stdv [unk]'},
                     'kdp': {'raw': 'Proc', 'default': False,
                             'tpkey': 'KDP [deg/km]'},
                     'ufvradh': {'raw': False, 'default': False,
                                 'tpkey': 'ufvradh [m/s]'},
                     'wradh': {'raw': 'Proc', 'default': False,
                               'tpkey': 'W [m/s]'},
                     'uwradh': {'raw': True, 'default': False,
                                'tpkey': 'W [m/s]'},
                     'uwradv': {'raw': True, 'default': False,
                                'tpkey': 'WV [m/s]'},
                     }
        rfile_path = self.file_name
        if get_rvar == 'all':
            radvars = {k: v for k, v in orvarname.items()}
        elif get_rvar == 'pvars':
            radvars = {k: v for k, v in orvarname.items()
                       if v['default']}
            if get_rawvars:
                radvars = {k: v['tpkey'] for k, v in radvars.items()
                           if v['raw'] is not False}
            else:
                radvars = {k: v['tpkey'] for k, v in radvars.items()
                           if v['raw'] is not True}
                radvars['uphidp'] = orvarname['uphidp']['tpkey']
                # warnings.warn('Warning: phidp not in data file,'
                #       'it will be replaced by uphidp')
        else:
            radvars = {k: v['tpkey'] for k, v in orvarname.items()
                       if k in get_rvar}
        if isinstance(rfile_path, str):
            if 'sweeph5onem_' in rfile_path:
                # prefix identifier within the file name
                kw_idf = 'sweeph5onem_'
                rvarf = rfile_path[rfile_path.find(kw_idf)
                                   + len(kw_idf):rfile_path.find('_0')]
                rvarr = orvarname.get(rvarf)['tpkey']
                rf2r = [rfile_path]
        elif isinstance(rfile_path, list):
            if 'sweeph5onem_' in rfile_path[0]:
                kw_idf = 'sweeph5onem_'
                rvarf = [i[i.find(kw_idf)+len(kw_idf):i.find('_0')]
                         for i in rfile_path]
                rvarr = [radvars.get(i) for i in rvarf]
                rvari = [c for c, v in enumerate(rvarr) if v is not None]
                rf2r = [rfile_path[i] for i in rvari]

        if isinstance(rfile_path, str) and 'sweeph5allm_' in rfile_path:
            kw_idf = 'sweeph5allm_'
            rf2r = [rfile_path]
        # =====================================================================
        # FILL UP TOWERPY OBJECT USING WR METHODS
        # =====================================================================
        # =====================================================================
        # Read and sort variables according to towerpy reqs.
        # =====================================================================
        # Using wrl>v2
        wrobjo = [xr.open_dataset(rf, engine='odim') for rf in rf2r]
        [wo.wrl.georef.georeference(crs=wrl.georef.get_default_projection())
         for wo in wrobjo]

        if kw_idf == 'sweeph5onem_':
            wrobj = {[k for k, v in wo.items()][0]: wo for wo in wrobjo}
            radvars = {k.upper(): v['tpkey'] for k, v in orvarname.items()
                       if k.upper() in wrobj.keys()}
            rvars = {}
            for k, v in radvars.items():
                rvars[v] = wrobj[k][k].values.astype(np.float64)
        # # 90 deg
        if kw_idf == 'sweeph5allm_':
            # wrobj = {[k for k, v in wo.items()][0]: wo for wo in wrobjo}
            rvars = {}
            for k, v in radvars.items():
                rvars[v] = wrobjo[0][k.upper()].values.astype(np.float64)
        # =====================================================================
        # Read and fill in parameters accordingly.
        # =====================================================================
        wrobj_params = wrl.io.read_generic_hdf5(rf2r[0])
        rparams = {'nvars': len(radvars)}
        rparams.update(wrobj_params['dataset1/what']['attrs'])
        rparams.update(wrobj_params['dataset1/how']['attrs'])
        rparams.update(wrobj_params['dataset1/where']['attrs'])
        rparams.update(wrobj_params['how']['attrs'])
        try:
            rparams.update(wrobj_params['how/monitor']['attrs'])
        except Exception:
            rparams.update(wrobj_params['how/monitoring']['attrs'])
            rparams.update(wrobj_params['how/monitoring/param']['attrs'])
            pass
        try:
            rparams.update(wrobj_params['how/radar_system']['attrs'])
        except Exception:
            rparams.update(wrobj_params['how/radarsystem/param']['attrs'])
            rparams.update(wrobj_params['dataset1/how/scanval/param']['attrs'])
            rparams.update(
                wrobj_params['dataset1/how/scanval/setting']['attrs'])
            pass
        rparams.update(wrobj_params['where']['attrs'])
        # Some params are encoded, so here those params are decoded
        rparams = dict((k, v.decode("utf-8")) if isinstance(v, np.bytes_)
                       else (k, v) for k, v in rparams.items())
        scandt = dt.datetime.strptime(rparams['startdate']
                                      + rparams['starttime'], '%Y%m%d%H%M%S%f')
        rparams['datetime'] = scandt.replace(tzinfo=ZoneInfo(tz))
        dtarrayx = list(rparams['datetime'].timetuple())[: -3]
        rparams['datetimearray'] = dtarrayx
        rparams['latitude [dd]'] = rparams.pop('lat')
        rparams['longitude [dd]'] = rparams.pop('lon')
        rparams['altitude [m]'] = rparams.pop('height')
        if isinstance(rfile_path, list):
            if all('pcpng' in fn for fn in self.file_name):
                rparams['elev_ang [deg]'] = 'precip_scan'
            else:
                rparams['elev_ang [deg]'] = rparams.pop('elangle')
        else:
            if 'pcpng' in self.file_name:
                rparams['elev_ang [deg]'] = 'precip_scan'
            else:
                rparams['elev_ang [deg]'] = rparams.pop('elangle')
        rparams['gateres [m]'] = rparams.pop('rscale')
        rparams['range_start [m]'] = rparams.pop('rstart')/1000
        rparams['ngates'] = rparams.pop('nbins')
        rparams['wavelength [cm]'] = rparams.pop('wavelength')
        rparams['beamwidth [deg]'] = rparams.pop('beamwidth')
        rparams['frequency [GHz]'] = (
            sc.c / rparams['wavelength [cm]'])/10000000
        rparams['phidp-offset_system [deg]'] = rparams.pop(
            'phidp-offset_system')
        rparams['prf [Hz]'] = rparams.pop('prf')
        rparams['pulselength [ms]'] = rparams.pop('pulsewidth')
        if 'radconstH' in rparams.keys():
            rparams['radar constant [dB]'] = rparams.pop('radconstH')
        else:
            rparams['radar constant [dB]'] = 0
        rparams['max_range [km]'] = rparams.pop('range')/1000
        if 'antgainH' in rparams.keys():
            rparams['antgainH [dB]'] = rparams.pop('antgainH')
        if 'antgainV' in rparams.keys():
            rparams['antgainV [dB]'] = rparams.pop('antgainV')
        rparams['Unambiguous velocity [m/s]'] = rparams.pop('NI')
        rparams['site_name'] = self.site_name.lower()

        # =====================================================================
        # Create a georeference grid for the DWD PPI radar scans.
        # =====================================================================
        if kw_idf == 'sweeph5onem_':
            igetp = list(wrobj.keys())[0]
            elev = np.deg2rad(wrobj[igetp].elevation.values)
            azim = np.deg2rad(wrobj[igetp].azimuth.values)
        elif kw_idf == 'sweeph5allm_':
            igetp = 0
            elev = np.deg2rad(wrobjo[igetp].elevation.values)
            azim = np.deg2rad(wrobjo[igetp].azimuth.values)

        gatei = rparams['range_start [m]']
        rng = np.arange(gatei,
                        rparams['ngates']*rparams['gateres [m]'],
                        rparams['gateres [m]'], dtype=float)
        rh, th = np.meshgrid(rng/1000, azim)
        bhkm = np.array([geo.height_beamc(ray, rng/1000)
                         for ray in np.rad2deg(elev)])
        bbhkm = np.array([geo.height_beamc(ray-rparams['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])
        bthkm = np.array([geo.height_beamc(ray+rparams['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])
        s = np.array([geo.cartesian_distance(ray, rng/1000, bhkm[0])
                      for i, ray in enumerate(np.rad2deg(elev))])
        a = [geo.pol2cart(arcl, azim) for arcl in s.T]
        xgrid = np.array([i[1] for i in a]).T
        ygrid = np.array([i[0] for i in a]).T

        geogrid = {'azim [rad]': azim,
                   'elev [rad]': elev,
                   'range [m]': rng,
                   'rho': rh,
                   'theta': th,
                   'grid_rectx': xgrid,
                   'grid_recty': ygrid}

        geogrid['beam_height [km]'] = bhkm
        geogrid['beambottom_height [km]'] = bbhkm
        geogrid['beamtop_height [km]'] = bthkm
        if kw_idf == 'sweeph5onem_':
            geogrid['grid_wgs84x'] = wrobj[igetp].x.values
            geogrid['grid_wgs84y'] = wrobj[igetp].y.values
        elif kw_idf == 'sweeph5allm_':
            geogrid['grid_wgs84x'] = wrobjo[igetp].x.values
            geogrid['grid_wgs84y'] = wrobjo[igetp].y.values

        proj_utm = wrl.georef.epsg_to_osr(32632)
        utm_coord = wrl.georef.spherical_to_centroids(wrobjo[0], crs=proj_utm)
        geogrid['grid_utmx'] = utm_coord[:, :, 0].values
        geogrid['grid_utmy'] = utm_coord[:, :, 1].values

        self.georef = geogrid
        self.params = rparams
        self.vars = rvars
        self.elev_angle = rparams['elev_ang [deg]']
        self.scandatetime = rparams['datetime']
        self.file_name = rf2r

    def ppi_dwd_polara(self, get_rvar='pvars', get_rawvars=False,
                       tz='Europe/Berlin'):
        """
        Read polara processed radar data computed.

        Parameters
        ----------
        get_rvar : list or string, optional
            Specifies the radar variables to read as a string or as a list
            of strings specifying the variables within the file to read.
            The string has to be one of ‘pvars’, ‘all’, or list using the
            identifiers specified in [1]_. The default is 'pvars'.
        get_rawvars : bool, optional
            Specifies if the radar variables are retrieved as
            filtered/processed (False) or uncorrected/raw (True).
            The default is False.
        tz : str, optional
            Key/name of the radar data timezone. The given tz value is then
            retrieved from the ZoneInfo module. The default is 'Europe/Berlin'.

        Returns
        -------
        Towerpy radar object.

        """
        orvarname = {'RA_AttCorrZhCorr': {'raw': 'Proc', 'default': False,
                                          'tpkey': 'ZH [dBZ]'},
                     'RA_AttCorrZDRCorr': {'raw': 'Proc', 'default': False,
                                           'tpkey': 'ZDR [dB]'},
                     'RA_PHIDP6': {'raw': 'Proc', 'default': False,
                                   'tpkey': 'PhiDP [deg]'},
                     'RA_RHV_C': {'raw': 'Proc', 'default': False,
                                  'tpkey': 'rhoHV [-]'},
                     'RA_AH': {'raw': 'Proc', 'default': False,
                               'tpkey': 'AH [dB/km]'},
                     'KDP_L': {'raw': 'Proc', 'default': False,
                               'tpkey': 'KDP [deg/km]'},
                     'ML_MAP': {'raw': 'Proc', 'default': False,
                                'tpkey': 'ML [-]'},
                     # 'IWC': {'raw': 'Proc', 'default': False,
                     #         'tpkey': 'IWC [gm-3]'},
                     # 'LWC': {'raw': 'Proc', 'default': False,
                     #         'tpkey': 'LWC [gm-3]'},
                     # 'RA_RAHKDP': {'raw': 'Proc', 'default': False,
                     #            'tpkey': 'Rainfall [mm/h]'}
                     }
        rfile_path = self.file_name
        if get_rvar == 'all':
            radvars = {k: v for k, v in orvarname.items()}
        elif get_rvar == 'pvars':
            radvars = {k: v for k, v in orvarname.items()
                       if v['default']}
            if get_rawvars:
                radvars = {k: v['tpkey'] for k, v in radvars.items()
                           if v['raw'] is not False}
            else:
                radvars = {k: v['tpkey'] for k, v in radvars.items()
                           if v['raw'] is not True}
                radvars['uphidp'] = orvarname['uphidp']['tpkey']
        else:
            radvars = {k: v['tpkey'] for k, v in orvarname.items()
                       if k in get_rvar}
        if isinstance(rfile_path, str):
            if 'sweeph5onem_' in rfile_path:
                # prefix identifier within the file name
                kw_idf = 'sweeph5onem_'
                rvarf = rfile_path[rfile_path.find(kw_idf)
                                   + len(kw_idf):rfile_path.find('_0')]
                rvarr = orvarname.get(rvarf)['tpkey']
                rf2r = [rfile_path]
        elif isinstance(rfile_path, list):
            if 'sweeph5onem_' in rfile_path[0]:
                kw_idf = 'sweeph5onem_'
                rvarf = [i[i.find(kw_idf)+len(kw_idf):i.find('_0')]
                         for i in rfile_path]
                rvarr = [radvars.get(i) for i in rvarf]
                rvari = [c for c, v in enumerate(rvarr) if v is not None]
                rf2r = [rfile_path[i] for i in rvari]
        if isinstance(rfile_path, str) and 'sweeph5allm_' in rfile_path:
            kw_idf = 'sweeph5allm_'
            rf2r = [rfile_path]
        # =====================================================================
        # FILL UP TOWERPY OBJECT USING WR METHODS
        # =====================================================================
        # =====================================================================
        # Read and sort variables according to towerpy reqs.
        # =====================================================================
        # Using wrl>v2
        wrobjo = [xr.open_dataset(rf, engine='odim') for rf in rf2r]
        [wo.wrl.georef.georeference(crs=wrl.georef.get_default_projection())
         for wo in wrobjo]

        if kw_idf == 'sweeph5onem_':
            wrobj = {[k for k, v in wo.items()][0]: wo for wo in wrobjo}
            radvars = {k: v['tpkey'] for k, v in orvarname.items()
                       if k in wrobj.keys()}
            rvars = {}
            for k, v in radvars.items():
                rvars[v] = wrobj[k][k].values.astype(np.float64)
        # # 90 deg
        if kw_idf == 'sweeph5allm_':
            # wrobj = {[k for k, v in wo.items()][0]: wo for wo in wrobjo}
            rvars = {}
            for k, v in radvars.items():
                rvars[v] = wrobjo[0][k.upper()].values.astype(np.float64)
        # =====================================================================
        # Read and fill in parameters accordingly.
        # =====================================================================
        wrobj_params = wrl.io.read_generic_hdf5(rf2r[0])
        rparams = {'nvars': len(radvars)}
        rparams.update(wrobj_params['dataset1/what']['attrs'])
        rparams.update(wrobj_params['dataset1/how']['attrs'])
        rparams.update(wrobj_params['dataset1/where']['attrs'])
        rparams.update(wrobj_params['how']['attrs'])
        try:
            rparams.update(wrobj_params['how/monitor']['attrs'])
        except Exception:
            rparams.update(wrobj_params['how/monitoring']['attrs'])
            rparams.update(wrobj_params['how/monitoring/param']['attrs'])
            pass
        try:
            rparams.update(wrobj_params['how/radar_system']['attrs'])
        except Exception:
            rparams.update(wrobj_params['how/radarsystem/param']['attrs'])
            rparams.update(wrobj_params['dataset1/how/scanval/param']['attrs'])
            rparams.update(
                wrobj_params['dataset1/how/scanval/setting']['attrs'])
            pass
        rparams.update(wrobj_params['where']['attrs'])
        # Some params are encoded, so here those params are decoded
        rparams = dict((k, v.decode("utf-8")) if isinstance(v, np.bytes_)
                       else (k, v) for k, v in rparams.items())
        scandt = dt.datetime.strptime(rparams['startdate']
                                      + rparams['starttime'], '%Y%m%d%H%M%S%f')
        rparams['datetime'] = scandt.replace(tzinfo=ZoneInfo(tz))
        dtarrayx = list(rparams['datetime'].timetuple())[: -3]
        rparams['datetimearray'] = dtarrayx
        rparams['latitude [dd]'] = rparams.pop('lat')
        rparams['longitude [dd]'] = rparams.pop('lon')
        rparams['altitude [m]'] = rparams.pop('height')
        if isinstance(rfile_path, list):
            if all('pcpng' in fn for fn in self.file_name):
                rparams['elev_ang [deg]'] = 'precip_scan'
            else:
                rparams['elev_ang [deg]'] = rparams.pop('elangle')
        else:
            if 'pcpng' in self.file_name:
                rparams['elev_ang [deg]'] = 'precip_scan'
            else:
                rparams['elev_ang [deg]'] = rparams.pop('elangle')
        rparams['gateres [m]'] = rparams.pop('rscale')
        rparams['range_start [m]'] = rparams.pop('rstart')/1000
        rparams['ngates'] = rparams.pop('nbins')
        rparams['wavelength [cm]'] = rparams.pop('wavelength')
        rparams['beamwidth [deg]'] = rparams.pop('beamwidth')
        rparams['frequency [GHz]'] = (
            sc.c / rparams['wavelength [cm]'])/10000000
        rparams['phidp-offset_system [deg]'] = rparams.pop(
            'phidp-offset_system')
        rparams['prf [Hz]'] = rparams.pop('prf')
        rparams['pulselength [ms]'] = rparams.pop('pulsewidth')
        if 'radconstH' in rparams.keys():
            rparams['radar constant [dB]'] = rparams.pop('radconstH')
        else:
            rparams['radar constant [dB]'] = 0
        rparams['max_range [km]'] = rparams.pop('range')/1000
        if 'antgainH' in rparams.keys():
            rparams['antgainH [dB]'] = rparams.pop('antgainH')
        if 'antgainV' in rparams.keys():
            rparams['antgainV [dB]'] = rparams.pop('antgainV')
        rparams['Unambiguous velocity [m/s]'] = rparams.pop('NI')
        rparams['site_name'] = self.site_name.lower()
        # =====================================================================
        # Create a georeference grid for the DWD PPI radar scans.
        # =====================================================================
        if kw_idf == 'sweeph5onem_':
            igetp = list(wrobj.keys())[0]
            elev = np.deg2rad(wrobj[igetp].elevation.values)
            azim = np.deg2rad(wrobj[igetp].azimuth.values)
        elif kw_idf == 'sweeph5allm_':
            igetp = 0
            elev = np.deg2rad(wrobjo[igetp].elevation.values)
            azim = np.deg2rad(wrobjo[igetp].azimuth.values)
        gatei = rparams['range_start [m]']
        rng = np.arange(gatei,
                        rparams['ngates']*rparams['gateres [m]'],
                        rparams['gateres [m]'], dtype=float)
        rh, th = np.meshgrid(rng/1000, azim)
        bhkm = np.array([geo.height_beamc(ray, rng/1000)
                         for ray in np.rad2deg(elev)])
        bbhkm = np.array([geo.height_beamc(ray-rparams['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])
        bthkm = np.array([geo.height_beamc(ray+rparams['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])
        s = np.array([geo.cartesian_distance(ray, rng/1000, bhkm[0])
                      for i, ray in enumerate(np.rad2deg(elev))])
        a = [geo.pol2cart(arcl, azim) for arcl in s.T]
        xgrid = np.array([i[1] for i in a]).T
        ygrid = np.array([i[0] for i in a]).T
        geogrid = {'azim [rad]': azim,
                   'elev [rad]': elev,
                   'range [m]': rng,
                   'rho': rh,
                   'theta': th,
                   'grid_rectx': xgrid,
                   'grid_recty': ygrid}
        geogrid['beam_height [km]'] = bhkm
        geogrid['beambottom_height [km]'] = bbhkm
        geogrid['beamtop_height [km]'] = bthkm
        if kw_idf == 'sweeph5onem_':
            geogrid['grid_wgs84x'] = wrobj[igetp].x.values
            geogrid['grid_wgs84y'] = wrobj[igetp].y.values
        elif kw_idf == 'sweeph5allm_':
            geogrid['grid_wgs84x'] = wrobjo[igetp].x.values
            geogrid['grid_wgs84y'] = wrobjo[igetp].y.values
        proj_utm = wrl.georef.epsg_to_osr(32632)
        utm_coord = wrl.georef.spherical_to_centroids(wrobjo[0], crs=proj_utm)
        geogrid['grid_utmx'] = utm_coord[:, :, 0].values
        geogrid['grid_utmy'] = utm_coord[:, :, 1].values
        self.georef = geogrid
        self.params = rparams
        self.vars = rvars
        self.elev_angle = rparams['elev_ang [deg]']
        self.scandatetime = rparams['datetime']
        self.file_name = rf2r


class RainGauge:
    """A Towerpy class to store rain gauge data."""

    def __init__(self, wdir, nwk_opr=None):
        """
        Init RainGauge Class with a filename.

        Parameters
        ----------
        wdir : string
            Path file containing the weather station data.
        nwk_opr : string
            Operator of the weather station network.
        """
        self.rg_wdir = wdir
        self.network_operator = nwk_opr

    def get_ukmidas_stn_mdata(self, midasmd, plot_methods=False):
        """
        Read metadata of UKMIDAS rain gauges from a csv/txt file.

        Parameters
        ----------
        midasmd : str
            Document containing descriptors of MIDAS rain gauges [1]_, such as
            latitude, longitude, stations name and ID, etc. Must be a
            csv or txt file using comma to separate values.

        References
        ----------
        .. [1]  Met Office (2006): MIDAS UK Hourly Rainfall Data. NCAS British
            Atmospheric Data Centre, 2024.
            https://catalogue.ceda.ac.uk/uuid/bbd6916225e7475514e17fdbf11141c1

        Returns
        -------
        midas_stn_mdata : dict
            Metadata of MIDAS rain gauges.
        """
        midas_rgdata = []
        with open(midasmd, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                midas_rgdata.append(row)
        # =============================================================================
        # Format metadata
        # =============================================================================
        midas_rgdidxi = [cnt for cnt, drw in enumerate(midas_rgdata)
                         if drw[0] == 'data']
        # or nr[0]=='end data']
        # midas_rgdescr = [drw for cnt, drw in enumerate(midas_rgdata)
        #                  if cnt <= midas_rgdidxi[0]]
        midas_rgsts = [drw for cnt, drw in enumerate(midas_rgdata)
                       if cnt > midas_rgdidxi[0]]
        midas_header = [kcols for kcols in midas_rgsts[0]]
        midas_stations = [i for i in midas_rgsts if i[0] != 'end data'][1:]
        dmmy = [{k: i[c] for c, k in enumerate(midas_header)}
                for i in midas_stations]
        midas_rgmd = {}
        for k in midas_header:
            midas_rgmd[k] = [d[k] for d in dmmy]
        # Format values
        midas_rgmd['first_year'] = np.asarray(midas_rgmd['first_year'],
                                              dtype=int)
        midas_rgmd['last_year'] = np.asarray(midas_rgmd['last_year'],
                                             dtype=int)
        midas_rgmd['src_id'] = np.asarray(midas_rgmd['src_id'], dtype=int)
        midas_rgmd['station_id'] = midas_rgmd.pop('src_id')
        midas_rgmd['station_latitude'] = np.asarray(
            midas_rgmd['station_latitude'], dtype=float)
        midas_rgmd['latitude [dd]'] = midas_rgmd.pop('station_latitude')
        midas_rgmd['station_longitude'] = np.asarray(
            midas_rgmd['station_longitude'], dtype=float)
        midas_rgmd['longitude [dd]'] = midas_rgmd.pop('station_longitude')
        midas_rgmd['station_elevation'] = np.asarray(
            midas_rgmd['station_elevation'], dtype=float)
        midas_rgmd['altitude [m]'] = midas_rgmd.pop('station_elevation')
        self.midas_stn_mdata = midas_rgmd
        if plot_methods:
            lpv = {'Altitude [m]':
                   [round(min(midas_rgmd['altitude [m]']), -2),
                    round(max(midas_rgmd['altitude [m]']), -2), 25]}
            bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
                   for key, value in lpv.items()}
            dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
                value, mpl.colormaps['gist_earth_r'].N, extend='max')
                      for key, value in bnd.items()}
            fig, ax1 = plt.subplots(figsize=(8, 8))
            mappable = ax1.scatter(midas_rgmd['longitude [dd]'],
                                   midas_rgmd['latitude [dd]'], marker='o',
                                   s=75, norm=dnorm['nAltitude [m]'],
                                   edgecolors='k',
                                   c=midas_rgmd['altitude [m]'],
                                   cmap=mpl.colormaps['gist_earth_r'],
                                   label=midas_rgmd['station_name'])
            ax1_divider = make_axes_locatable(ax1)
            cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=12)
            plt.title('altitude [m]', fontsize=14)
            cax1.xaxis.set_ticks_position('top')
            ax1.grid(True)
            ax1.axis('tight')
            ax1.set_aspect("equal")
            fig.suptitle(f'Weather stations from {self.network_operator}',
                         fontsize=16)
            plt.tight_layout()

    def get_dwdstn_mdata(self, rgmd_fname, plot_methods=False):
        """
        Read metadata of DWD rain gauges from a csv/txt file.

        Parameters
        ----------
        rgmd_fname : str
            Document containing descriptors of DWD rain gauges, such as
            latitude, longitude, stations name and ID, etc. Must be a
            csv or txt file containing comma-separated values.

        References
        ----------
        .. [1] Deutscher Wetterdienst, 2023. Hourly station observations of
            precipitation for Germany (v23.3) Retrieved from
            https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf

        Returns
        -------
        dwd_stn_mdata : dict
            Metadata of DWD rain gauges.
        """
        with open(rgmd_fname) as f:
            lines = f.readlines()
        dwd_stations = [i[:-1].split(',') for i in lines]
        dwd_header = [kcols for kcols in dwd_stations[0]]
        dmmy = [{k: i[c] for c, k in enumerate(dwd_header)}
                for i in dwd_stations][1:]
        dwd_rgmd = {}
        for k in dwd_header:
            dwd_rgmd[k] = [d[k] for d in dmmy]
        # Format values
        dwd_rgmd['geoBreite'] = np.asarray(dwd_rgmd['geoBreite'], dtype=float)
        dwd_rgmd['latitude [dd]'] = dwd_rgmd.pop('geoBreite')
        dwd_rgmd['geoLaenge'] = np.asarray(dwd_rgmd['geoLaenge'], dtype=float)
        dwd_rgmd['longitude [dd]'] = dwd_rgmd.pop('geoLaenge')
        dwd_rgmd['Stations_id'] = np.asarray(dwd_rgmd['Stations_id'],
                                             dtype=int)
        dwd_rgmd['station_id'] = dwd_rgmd.pop('Stations_id')
        dwd_rgmd['stations_name'] = dwd_rgmd.pop('Stationsname')
        dwd_rgmd['Stationshoehe'] = np.asarray(dwd_rgmd['Stationshoehe'],
                                               dtype=float)
        dwd_rgmd['altitude [m]'] = dwd_rgmd.pop('Stationshoehe')
        ftime_date = [dt.datetime.strptime(i, '%Y%m%d')
                      for i in dwd_rgmd['bis_datum']]
        itime_date = [dt.datetime.strptime(i, '%Y%m%d')
                      for i in dwd_rgmd['von_datum']]
        dwd_rgmd['temporal_coverage'] = [(i, j) for i, j
                                         in zip(itime_date, ftime_date)]
        dwd_rgmd.pop('bis_datum')
        dwd_rgmd.pop('von_datum')
        if plot_methods:
            lpv = {'Altitude [m]':
                   [round(min(dwd_rgmd['altitude [m]']), -2),
                    round(max(dwd_rgmd['altitude [m]']), -2), 25]}
            bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
                   for key, value in lpv.items()}
            dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
                value,
                mpl.colormaps['gist_earth_r'].N,
                extend='max')
                     for key, value in bnd.items()}

            fig, ax1 = plt.subplots(figsize=(8, 8))
            mappable = ax1.scatter(dwd_rgmd['longitude [dd]'],
                                   dwd_rgmd['latitude [dd]'],
                                   marker='o', s=75,
                                   norm=dnorm['nAltitude [m]'],
                                   edgecolors='k', c=dwd_rgmd['altitude [m]'],
                                   cmap=mpl.colormaps['gist_earth_r'],
                                   label=dwd_rgmd['stations_name'])
            ax1_divider = make_axes_locatable(ax1)
            cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=12)
            plt.title('altitude [m]', fontsize=14)
            cax1.xaxis.set_ticks_position('top')
            ax1.grid(True)
            ax1.axis('tight')
            ax1.set_aspect("equal")
            fig.suptitle(f'Weather stations from {self.network_operator}',
                         fontsize=16)
            plt.tight_layout()
        self.dwd_stn_mdata = dwd_rgmd

    def get_stns_rad(self, rad_georef, rad_params, rg_mdata, dmax2rad=50,
                     iradbins=100, dmax2radbin=1, del_by_station_id=None,
                     plot_methods=False):
        """
        Get weather stations near radar locations using a metadata list.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others. Must containg x- and y-grid of
            WGS84 coordinates (Lon/Lat).
        rad_params : dict
            Radar technical details.
        rg_mdata : dict
            Rain gauge metadata descriptors.
        dmax2rad : float, optional
            Maximum distance [in km] allowed between the radar and rain gauges.
            The default is 50.
        iradbins : int, optional
            Initial number of radar bins neighbouring the rain gauges to be
            gathered. The default is 100.
        dmax2radbin : float, optional
            Maximum distance [in km] allowed between each radar bin and the
            nearest rain gauge. The default is 1.
        del_by_station_id : list, optional
            Deletes stations based on their ID number. The default is None.
        plot_methods : bool, optional
            Plot the location of the radar, the rain gauges and the
            corresponding radar bins. The default is False.

        Returns
        -------
        stn_near_rad : dict

        """
        # Compute d[km] from radar to rain gauges
        disth = DistanceMetric.get_metric('haversine')
        dist_rd2rg = {c: 6378 * disth.pairwise([
            [np.radians(rad_params['latitude [dd]']),
             np.radians(rad_params['longitude [dd]'])],
            [np.radians(rg_mdata['latitude [dd]'][c]),
             np.radians(rg_mdata['longitude [dd]'][c])]]).item(1)
            for c, v in enumerate(rg_mdata['station_id'])}
        dist_rd2rgix = [k for k, v in dist_rd2rg.items() if v <= dmax2rad]
        rg_mdataf = {k: [v2 for c, v2 in enumerate(v) if c in dist_rd2rgix]
                     for k, v in rg_mdata.items()}
        rg_mdataf['distance2rad [km]'] = [dist_rd2rg[i] for i in dist_rd2rgix]
        # Get nearest radar bins for each rain gauge
        binx = rad_georef['grid_wgs84x'].ravel()
        biny = rad_georef['grid_wgs84y'].ravel()
        tree = spatial.KDTree(list(zip(binx, biny)))
        kdd, kdix = tree.query(list(zip(rg_mdataf['longitude [dd]'],
                                        rg_mdataf['latitude [dd]'])),
                               k=iradbins)
        binx_nn = binx[kdix]
        biny_nn = biny[kdix]
        # Compute distance from each rbin to rain gauges
        kddkm = np.array([np.array([6378 * disth.pairwise([
            [np.radians(biny_nn[c][c2]), np.radians(vbin)],
            [np.radians(rg_mdataf['latitude [dd]'][c]),
             np.radians(rg_mdataf['longitude [dd]'][c])]]).item(1)
            for c2, vbin in enumerate(v)]) for c, v in enumerate(binx_nn)])
        # Filter radarbins usign a distance threshhold
        kddkm[kddkm > dmax2radbin] = np.nan
        binx_nn[np.isnan(kddkm)] = np.nan
        biny_nn[np.isnan(kddkm)] = np.nan

        rg_mdataf['kd_rbin_idx'] = kdix
        rg_mdataf['kd_rbin_dkm'] = kddkm
        # Transform the lists within dict into np.array
        rg_mdataf = {k: (np.array(rg_mdataf[k])
                         if not any(isinstance(y, str)
                                    for y in rg_mdataf[k])
                         else rg_mdataf[k])
                     for k in rg_mdataf.keys()}
        if del_by_station_id:
            idx_id2rm = np.argwhere(
                rg_mdataf['station_id'].reshape(-1, 1)
                == del_by_station_id)[:, 0]
            rg_mdataf = {k: [dwdr for c, dwdr in enumerate(v)
                             if c not in idx_id2rm]
                         for k, v in rg_mdataf.items()}
            rg_mdataf = {k: (np.array(rg_mdataf[k])
                             if not any(isinstance(y, str)
                                        for y in rg_mdataf[k])
                             else rg_mdataf[k])
                         for k in rg_mdataf.keys()}
        if plot_methods:
            lpv = {'Altitude [m]':
                   [round(min(rg_mdata['altitude [m]']), -2),
                    round(max(rg_mdata['altitude [m]']), -2), 25]}
            bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
                   for key, value in lpv.items()}
            dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
                value, mpl.colormaps['gist_earth_r'].N, extend='max')
                     for key, value in bnd.items()}
            rdatah = {'Beam_height [km]':
                      rad_georef['beam_height [km]']
                      + rad_params['altitude [m]'] / 1000}
            rg_acprecip = {'grid_wgs84x': rg_mdataf['longitude [dd]'],
                           'grid_wgs84y': rg_mdataf['latitude [dd]'],
                           'altitude [km]':
                               rg_mdataf['altitude [m]'].flatten()/1000}
            dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(binx, biny, c='gray', marker='+', alpha=0.1,
                     linestyle='None')
            ax1.plot(binx_nn, biny_nn, 'k.')
            ax1.plot(rad_params['longitude [dd]'],
                     rad_params['latitude [dd]'], c='tab:orange', marker='X',
                     linestyle='None', label='Radar site')
            ax1.plot(rg_mdataf['longitude [dd]'], rg_mdataf['latitude [dd]'],
                     marker='x', linestyle='None', c='tab:grey', markersize=5)
            mappable = ax1.scatter(rg_mdata['longitude [dd]'],
                                   rg_mdata['latitude [dd]'], marker='o',
                                   s=75, norm=dnorm['nAltitude [m]'],
                                   edgecolors='k', c=rg_mdata['altitude [m]'],
                                   cmap=mpl.colormaps['gist_earth_r'])
            plt.plot(rad_georef['grid_wgs84x'][:, -1],
                     rad_georef['grid_wgs84y'][:, -1], 'gray')

            ax1_divider = make_axes_locatable(ax1)
            cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=12)
            plt.title('altitude [m]', fontsize=14)
            cax1.xaxis.set_ticks_position('top')
            ax1.grid(True)
            ax1.legend()
            ax1.axis('tight')
            ax1.set_aspect("equal")
            fig.suptitle(f'Weather stations from {self.network_operator}',
                         fontsize=16)
            plt.tight_layout()
            rad_display.plot_ppi(
                rad_georef, rad_params, rdatah,  points2plot=rg_acprecip,
                ptsvar2plot='altitude [km]', cpy_feats={'status': True},
                data_proj=ccrs.PlateCarree(), proj_suffix='wgs84',
                fig_title=f'Elevation angle: {dtdes1}', fig_size=(13, 7))
        self.stn_near_rad = rg_mdataf

    def get_stns_box(self, rg_mdata, bbox_xlims=[-np.inf, np.inf],
                     bbox_ylims=[-np.inf, np.inf],
                     bbox_zlims=[-np.inf, np.inf], surface=None, isrfbins=50,
                     dmax2srfbin=1, del_by_station_id=None,
                     plot_methods=False):
        """
        Get weather stations using coordinates as a bounding box.

        Parameters
        ----------
        rg_mdata : dict
            Rain gauge metadata descriptors.
        bbox_xlims : 2-element tuple or list, optional
            Min-max values in the longitude [dd] to create the bounding box.
        bbox_ylims : 2-element tuple or list, optional
            Min-max values in the latitude [dd] to create the bounding box.
        bbox_zlims : 2-element tuple or list, optional
            Min-max values in the altitude [m] to create the bounding box.
            The default is [-np.inf, np.inf].
        surface : dict, optional
            Georeferenced data containing the x- and y-grid of WGS84
            coordinates (Lon/Lat). The default is None.
        isrfbins : int, optional
            Initial number of surface bins neighbouring the rain gauges to be
            gathered. The default is 50.
        dmax2srfbin : float, optional
            Maximum distance [in km] allowed between each surface bin and the
            nearest rain gauge. The default is 1.
        del_by_station_id : list, optional
            Deletes stations based on their ID number. The default is None.
        plot_methods : bool, optional
            Plot the location of the radar, the rain gauges and the
            corresponding bounding box. The default is False.

        Returns
        -------
        stn_bbox : dict

        """
        # TODO: Surface should get own limits
        bound_x = np.logical_and(rg_mdata['longitude [dd]'] > bbox_xlims[0],
                                 rg_mdata['longitude [dd]'] < bbox_xlims[1])
        bound_y = np.logical_and(rg_mdata['latitude [dd]'] > bbox_ylims[0],
                                 rg_mdata['latitude [dd]'] < bbox_ylims[1])
        bound_z = np.logical_and(rg_mdata['altitude [m]'] > bbox_zlims[0],
                                 rg_mdata['altitude [m]'] < bbox_zlims[1])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        bbf_idx = [cnt for cnt, bbf in enumerate(bb_filter) if bbf]

        rg_mdataf = {k: [v2 for c, v2 in enumerate(v) if c in bbf_idx]
                     for k, v in rg_mdata.items()}
        rg_mdataf = {k: (np.array(rg_mdataf[k])
                         if not any(isinstance(y, str)
                                    for y in rg_mdataf[k])
                         else rg_mdataf[k])
                     for k in rg_mdataf.keys()}

        if surface:
            disth = DistanceMetric.get_metric('haversine')
            binx = surface['grid_wgs84x'].ravel()
            biny = surface['grid_wgs84y'].ravel()
            tree = spatial.KDTree(list(zip(binx, biny)))
            kdd, kdix = tree.query(list(zip(rg_mdataf['longitude [dd]'],
                                            rg_mdataf['latitude [dd]'])),
                                   k=isrfbins)
            binx_nn = binx[kdix]
            biny_nn = biny[kdix]
            # Compute distance from each rbin to rain gauges
            kddkm = np.array([np.array([6378 * disth.pairwise([
                [np.radians(biny_nn[c][c2]), np.radians(vbin)],
                [np.radians(rg_mdataf['latitude [dd]'][c]),
                 np.radians(rg_mdataf['longitude [dd]'][c])]]).item(1)
                for c2, vbin in enumerate(v)]) for c, v in enumerate(binx_nn)])
            # Filter radarbins usign a distance threshhold
            kddkm[kddkm > dmax2srfbin] = np.nan
            binx_nn[np.isnan(kddkm)] = np.nan
            biny_nn[np.isnan(kddkm)] = np.nan

            rg_mdataf['kd_rbin_idx'] = kdix
            rg_mdataf['kd_rbin_dkm'] = kddkm
            # Transform the lists within dict into np.array
            rg_mdataf = {k: (np.array(rg_mdataf[k])
                             if not any(isinstance(y, str)
                                        for y in rg_mdataf[k])
                             else rg_mdataf[k])
                         for k in rg_mdataf.keys()}

        if del_by_station_id:
            idx_id2rm = np.argwhere(
                rg_mdataf['station_id'].reshape(-1, 1)
                == del_by_station_id)[:, 0]
            rg_mdataf = {k: [dwdr for c, dwdr in enumerate(v)
                             if c not in idx_id2rm]
                         for k, v in rg_mdataf.items()}
            rg_mdataf = {k: (np.array(rg_mdataf[k])
                             if not any(isinstance(y, str)
                                        for y in rg_mdataf[k])
                             else rg_mdataf[k])
                         for k in rg_mdataf.keys()}

        if plot_methods:
            lpv = {'Altitude [m]':
                   [round(min(rg_mdata['altitude [m]']), -2),
                    round(max(rg_mdata['altitude [m]']), -2), 25]}
            bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
                   for key, value in lpv.items()}
            dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
                value, mpl.colormaps['gist_earth_r'].N, extend='max')
                     for key, value in bnd.items()}
            fig, ax1 = plt.subplots(figsize=(8, 8))
            if surface:
                ax1.plot(binx, biny, c='gray', marker='+', alpha=0.1,
                         linestyle='None')
                ax1.plot(binx_nn, biny_nn, 'k.')
            ax1.add_patch(Rectangle((bbox_xlims[0], bbox_ylims[0]),
                                    bbox_xlims[1]-bbox_xlims[0],
                                    bbox_ylims[1]-bbox_ylims[0], fc='w',
                                    ec='k', lw=3))
            mappable = ax1.scatter(rg_mdata['longitude [dd]'],
                                   rg_mdata['latitude [dd]'], marker='o', s=75,
                                   norm=dnorm['nAltitude [m]'],
                                   edgecolors='k', c=rg_mdata['altitude [m]'],
                                   cmap=mpl.colormaps['gist_earth_r'])
            ax1.plot(rg_mdataf['longitude [dd]'], rg_mdataf['latitude [dd]'],
                     marker='x', markersize=5, linestyle='None', c='k')

            ax1_divider = make_axes_locatable(ax1)
            cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=12)
            plt.title('altitude [m]', fontsize=14)
            cax1.xaxis.set_ticks_position('top')
            ax1.grid(True)
            ax1.axis('tight')
            ax1.set_aspect("equal")
            fig.suptitle(f'Weather stations from {self.network_operator}',
                         fontsize=16)
            plt.tight_layout()
        self.stn_bbox = rg_mdataf

    def get_dwdstn_nc(self, station_id, start_time, stop_time, dir_ncdf=None,
                      period='historical', resolution='hourly', reload=False):
        """
        Download Open-Acces rain gauge precipitation data using wetterdienst.

        Parameters
        ----------
        station_id : int
            Station identifier.
        start_time : datetime
            Datetime object. Initial date and time to download data.
        stop_time : datetime
            Datetime object. Final date and time to download data.
        dir_ncdf : str, optional
            Folder path to save downloaded data. If None uses the
            working directory (RainGauge.rg_wdir).
        period : str, optional
            Dataset to be recovered. The string has to be one of 'historical'
            or ‘recent’ according to [1]_. The default is 'historical'.
        resolution : str, optional
            Resolution of the dataset to be recovered. The string has to be
            one of dict(dwdres) according to [1]_. The default is 'hourly'.
        reload : bool, optional
            Overwrite existing files. The default is False.

        References
        ----------
        .. [1] Deutscher Wetterdienst, 2023. Hourly station observations of
            precipitation for Germany (v23.3) Retrieved from
        https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf

        """
        from wetterdienst import Wetterdienst, Resolution
        from wetterdienst.provider.dwd.observation import DwdObservationDataset
        from wetterdienst.provider.dwd.observation import DwdObservationPeriod
        from zipfile import BadZipFile

        print(f'processing station: {int(station_id):05d} \n'
              + f'{start_time:%Y-%m-%d¦%H:%M:%S} -- '
              + f'{stop_time:%Y-%m-%d¦%H:%M:%S}')
        outfile = (f'{int(station_id):05d}_{start_time:%Y%m%d%H%M%S}'
                   + f'_{stop_time:%Y%m%d%H%M%S}.nc')
        # print('processing station:' + outfile)
        # check if already exists
        if os.path.isfile(outfile) and reload is False:
            print(f'--- station {int(station_id):05d} already exists')
            return
        # used by DWD for file server
        dwdres = {'1_minute': Resolution.MINUTE_1,
                  '5_minutes': Resolution.MINUTE_5,
                  '10_minutes': Resolution.MINUTE_10,
                  '15_minutes': Resolution.MINUTE_15,
                  'hourly': Resolution.HOURLY, '6_hour': Resolution.HOUR_6,
                  'subdaily': Resolution.SUBDAILY, 'daily': Resolution.DAILY,
                  'monthly': Resolution.MONTHLY,
                  'annual': Resolution.ANNUAL,
                  # For sources without resolution
                  # 'undefined': UNDEFINED,
                  # 'dynamic': ResolutionType.DYNAMIC.value
                  }
        dataset_res = dwdres.get('hourly')
        API = Wetterdienst(provider='dwd', network='observation')
        station_id = [station_id]
        if period == 'historical':
            dataset_period = DwdObservationPeriod.HISTORICAL
        elif period == 'recent':
            dataset_period = DwdObservationPeriod.RECENT,
        stations = API(
            parameter=DwdObservationDataset.PRECIPITATION,
            resolution=dataset_res,
            period=dataset_period,
            start_date=start_time,
            end_date=stop_time,
        ).filter_by_station_id(station_id=station_id)
        results = stations.values.all()
        if results.df.shape != (0, 0):
            # rain = results.df.to_numpy()
            try:
                rain = results.df.to_pandas()
                rain = rain.to_xarray()
                # extract precipitation_height
                rain = rain.where(rain.parameter == 'precipitation_height',
                                  drop=True)
                # convert date to time coordinate (datetime64[ns])
                rain = rain.assign_coords(
                    dict(time=rain.date.astype('datetime64[ns]')))
                # swap index vs. time drop index
                rain = rain.swap_dims(dict(index='time')).drop(['index'])
                # retrieve station_id and value_name
                sid = list(set(rain.station_id.values))[0]
                value_name = list(set(rain.parameter.values))[0]
                # rename value
                rain = rain.rename_vars(dict(value=value_name))
                # set some attributes
                rain[value_name].attrs['short_name'] = value_name
                rain[value_name].attrs['unit'] = 'mm'
                # drop unnecessary variables
                rain = rain.drop_vars(['station_id', 'dataset', 'parameter',
                                       'date'])
                # assign station_id coordinate (merge several stations)
                rain = rain.assign_coords(station_id=sid)
                # extract time strings
                start_time_str = rain.time.dt.strftime(
                    '%Y%m%d%H%M%S')[0].values
                stop_time_str = rain.time.dt.strftime(
                    '%Y%m%d%H%M%S')[-1].values
                outfile = f'{int(sid):05d}_{start_time_str}_{stop_time_str}.nc'
                # print('--- save to ', outfile)
                if dir_ncdf is not None:
                    rain.to_netcdf(dir_ncdf+outfile)
                else:
                    rain.to_netcdf(self.rg_wdir+outfile)
            except Exception as err:
                if dir_ncdf is not None:
                    log_file = open(dir_ncdf+'log.txt', 'a')
                else:
                    log_file = open(self.rg_wdir+'log.txt', 'a')
                log_file.write(f'Error ({err}) in {outfile}-'+'\n')
                pass
            except BadZipFile as err:
                if dir_ncdf is not None:
                    log_file = open(dir_ncdf+'log.txt', 'a')
                else:
                    log_file = open(self.rg_wdir+'log.txt', 'a')
                log_file.write(f'Error ({err}) in {outfile}-'+'\n')
                pass
        else:
            if dir_ncdf is not None:
                log_file = open(dir_ncdf+'log.txt', 'a')
            else:
                log_file = open(self.rg_wdir+'log.txt', 'a')
            log_file.write(f'Error in {outfile}-'+'\n')

    def get_rgdata(self, dtime, ds2read=None, ds_ncdir=None,
                   ncfdtf='%Y%m%d%H%M%S', ds_tres=dt.timedelta(hours=1),
                   dt_fwd=dt.timedelta(hours=0),
                   dt_bkwd=dt.timedelta(hours=12),
                   ds_accum=dt.timedelta(hours=12),
                   ncf_tres=dt.timedelta(hours=1), sort_rgdata='default',
                   drop_nan=False, drop_thrb=False, rprod_fltr=None,
                   rprod_thr=None, plot_methods=False):
        """
        Retrieve and process rain-gauge precipitation data from nc files.

        Parameters
        ----------
        dtime : datetime
            Date and time of the datasets to retrieve.
        ds2read : dict
            Metadata of the rain gauges, describing location, IDNo.,
            among others. If None uses the results of RainGauge.get_stns_rad
        ds_ncdir : str, optional
            Folder path where the *.nc files are stored. If None uses the
            working directory (RainGauge.rg_wdir). File names format must be
            as 'stationID_STARTTIME_ENDTIME.nc'.
        ncfdtf : str, optional
            Format of STARTTIME and ENDTIME descriptors in the *.nc file name.
            The default is '%Y%m%d%H%M%S'.
        ds_tres : dt.timedelta object
            Temporal resolution of the station datasets. The default is
            dt.timedelta(hours=1).
        dt_fwd : dt.timedelta object
            Time to be added to dtime for retrieving datasets.
            The default is dt.timedelta(minutes=0).
        dt_bkwd : dt.timedelta object
            Time to be subtracted from dtime for retrieving datasets.
            The default is dt.timedelta(hours=24).
        ds_accum : dt.timedelta object
            Accumulates the precipitation in a given time window.
            The default is dt.timedelta(hours=1).
        ncf_tres : dt.timedelta object
            Temporal resolution of the *.nc files to be readed.
            The default is dt.timedelta(hours=1).
        sort_rgdata : dict, optional
            Additional parameters used to sort the output dictionary.
            The default is {'sort': False, 'dkey': 'rain_sum',
                            'order': 'descending'}.
        drop_nan : bool, optional
            If True, removes empty rain gauge datasets. The default is False.
        drop_thrb : int or float, optional
            If True, removes rain gauge dataset if all records are below a
            given threshold. The default is False.
        rprod_fltr : array, optional
            If given, use surface from rain product to remove
            rain gauge dataset if all records are below a given
            threshold (rprod_thr). The default is None.
        rprod_thr : int or float, optional
            Threshold use to remove rain gauge datasets that are below
            this value. The default is None.
        plot_methods : bool, optional
            Plots the processing of the weather station data.
            The default is False.

        Returns
        -------
        rg_precip : dict

        """
        if ds_ncdir is None:
            ds_ncdir = self.rg_wdir
        else:
            ds_ncdir = ds_ncdir
        if ds2read is None:
            ds2read = self.stn_near_rad
        else:
            ds2read = ds2read
        sortdict = {'sort': False, 'order': 'descending', 'dkey': 'rain_sum'}
        if sort_rgdata != 'default':
            sortdict.update(sort_rgdata)
        rg_precipO = dict(ds2read)
        rtime = dtime.replace(tzinfo=None)
        # Creates expected time stamps using the rain gauge data resolution.
        trg_full = np.arange(time_round(rtime - dt_bkwd, ds_tres) + ds_tres,
                             time_round(rtime + dt_fwd, ds_tres) + ds_tres,
                             ds_tres).astype(dt.datetime)
        # maybe it's better here to group by timesteps
        # ds_accumg = round((trg_full[-1]-trg_full[0]+ds_tres)/ds_accum)
        ds_accumtg = int(ds_accum/ds_tres)
        trg_fullg = list(zip_longest(*(iter(trg_full),) * ds_accumtg))
        trg_fullg = [[itm for itm in l1 if itm is not None]
                     for l1 in trg_fullg]
        # Read nc files within fdir
        rg_nc = [i for i in os.listdir(ds_ncdir) if i.endswith('.nc')]
        # Filter nearby stations using station ID
        rg_nc[:] = [ncf.removesuffix('.nc') for ncf in rg_nc
                    if any(stid == int(ncf[:ncf.find('_')])
                           for stid in rg_precipO['station_id'])]
        # Remove station ID
        rg_ncdt = [ncf[ncf.find('_') + 1:] for ncf in rg_nc]
        # Get start/end time of the file names as tuple
        rg_ncdt[:] = [(ncf[:ncf.find('_')], ncf[ncf.find('_') + 1:])
                      for ncf in rg_ncdt]
        # ncfdtf = ncfdtf
        rg_ncdt[:] = [[dt.datetime.strptime(tfdt, ncfdtf) for tfdt in fdt]
                      for fdt in rg_ncdt]
        # Filter nearby stations using timestamp in files name
        ncdtei = time_round(trg_full[0], ncf_tres)
        ncdtef = time_round(trg_full[-1] + ncf_tres, ncf_tres)
        rg_ncvdt = [i for i, d in enumerate(rg_ncdt)
                    if d[0] >= ncdtei and d[1] <= ncdtef]
        rg_ncv = [rg_nc[vidx] for vidx in rg_ncvdt]
        # Read-in nc rain gagues data
        rgd_rain = [netcdf2pydict(ds_ncdir+i+'.nc') for i in rg_ncv]
        # check if the dicts are from same station and combines them
        rgd_rain2 = [[j for j in rgd_rain if j['station_id'] == i]
                     for i in list(sorted(set([stid['station_id']
                                               for stid in rgd_rain])))]
        # Get keys of all dicts
        rgd_rainfk = list(set([k for rgd in rgd_rain2 for rgdk in rgd
                          for k in rgdk.keys()]))
        rgd_rainc = [{k: np.hstack([rgh[k] for rgh in rgn])
                      for k in rgd_rainfk} for rgn in rgd_rain2]
        idx_vtime = [[[[c for c, tf in enumerate(rg['time']) if tf == t][0]
                       if t in rg['time'] else np.nan for t in tg]
                      for tg in trg_fullg]
                     for rg in rgd_rainc]
        # Retrieve metadata using station_id
        stns_id = np.array([int(i3['station_id'][0])
                            for c, i3 in enumerate(rgd_rainc)])
        idxvids = np.array([np.argwhere(rg_precipO['station_id'] == i)[0]
                            for i in stns_id])
        rg_precip = {k: (v[idxvids] if isinstance(v, np.ndarray)
                         else [i for c, i in enumerate(rg_precipO[k])
                               if c in idxvids])
                     for k, v in rg_precipO.items()}
        # Gather rain rates on the original resolution
        rg_precip['rain_ires'] = [np.array([
            [rgd_rainc[c]['precipitation_height [mm]'][i]
             for i in i2 if isinstance(i, int)]
            for i2 in i3])
            for c, i3 in enumerate(idx_vtime)]
        # Timestamps of the rain rates
        rg_precip['rain_idt'] = [[[rgd_rainc[c]['time'][i]
                                   for i in i2 if isinstance(i, int)]
                                  for i2 in i3]
                                 for c, i3 in enumerate(idx_vtime)]
        # Derive rain rate acummulations for a given delta time
        rg_precip['rain_sum'] = [np.array([np.nansum(j) for j in i])
                                 for i in rg_precip['rain_ires']]
        # TODO Here maybe it is possible to get last value and then computecum
        rg_precip['rain_cumsum'] = [np.array([np.nancumsum(j) for j in i])
                                    for i in rg_precip['rain_ires']]
        # Filter nanacums
        for dtm, dmk in zip(rg_precip['rain_cumsum'], rg_precip['rain_ires']):
            dtm[np.isnan(dmk)] = np.nan
        # Timestamps of the rain rates
        if np.isnan(idx_vtime).all():
            rg_precip['rain_sumdt'] = []
        else:
            rg_precip['rain_sumdt'] = [[[rgd_rainc[c]['time'][i] for i in i2
                                         if isinstance(i, int)][-1]
                                        for i2 in i3]
                                       for c, i3 in enumerate(idx_vtime)]
        if drop_thrb is not None:
            vidx_thr = [c for c, dwdr in enumerate(rg_precip['rain_sum'])
                        if not all(dwdr.flatten() <= drop_thrb)]
            rg_precip = {k: [dwdr for c, dwdr in enumerate(v) if c in vidx_thr]
                         for k, v in rg_precip.items()}
            rg_precip['kd_rbin_idx'] = np.array(rg_precip['kd_rbin_idx'])
            rg_precip['kd_rbin_dkm'] = np.array(rg_precip['kd_rbin_dkm'])
            # rlog = [np.log10(i[0]*12) for i in dwd_rgf_rsum]
            # rlog = np.log10(rsum*12)
        if drop_nan is not None:
            vidx_nan = [c for c, dwdr in enumerate(rg_precip['rain_ires'])
                        if not np.isnan(dwdr).all()]
            rg_precip = {k: [dwdr for c, dwdr in enumerate(v) if c in vidx_nan]
                         for k, v in rg_precip.items()}
            rg_precip['kd_rbin_idx'] = np.array(rg_precip['kd_rbin_idx'])
            rg_precip['kd_rbin_dkm'] = np.array(rg_precip['kd_rbin_dkm'])
            # rlog = [np.log10(i[0]*12) for i in dwd_rgf_rsum]
            # rlog = np.log10(rsum*12)
        # Filter using rprod_fltr and drop_thrb
        if rprod_fltr is not None and rprod_thr is not None:
            vidx_rp = [iix for iix, ivl in
                       enumerate(rprod_fltr.flatten()
                                 [rg_precip['kd_rbin_idx']])
                       if np.nanmean(ivl) > rprod_thr
                       # and np.count_nonzero(~np.isnan(ivl)) > 8
                       ]
            rg_precip = {k: [dwdr for c, dwdr in enumerate(v) if c in vidx_rp]
                         for k, v in rg_precip.items()}
            rg_precip['kd_rbin_idx'] = np.array(rg_precip['kd_rbin_idx'])
            rg_precip['kd_rbin_dkm'] = np.array(rg_precip['kd_rbin_dkm'])
        if sortdict['sort']:
            if sortdict['order'] == 'ascending':
                sord = False
            elif sortdict['order'] == 'descending':
                sord = True
            vsort = sorted([(v, c) for c, v in enumerate(
                np.array(rg_precip[sortdict['dkey']])
                .flatten().argsort().argsort())], reverse=sord)
            vsort[:] = [i[1] for i in vsort]
            rg_precip = {k: [v[c] for c in vsort]
                         for k, v in rg_precip.items()}
        # Transform the lists within dict into np.array
        rg_precip = {k: (np.array(rg_precip[k])
                         if not any(isinstance(y, str)
                                    for y in rg_precip[k])
                         else rg_precip[k])
                     for k in rg_precip.keys()}
        if plot_methods:
            maxplt = 12
            nitems = len(rg_precip['station_id'])
            nplots = [[i*maxplt, (1+i)*maxplt]
                      for i in range(int(np.ceil(nitems/maxplt)))]
            nplots[-1][-1] = nitems
            if nitems > maxplt:
                nitems = maxplt
            ncols = int(nitems**0.5)
            nrows = nitems // ncols
            # Number of rows, add one if necessary
            if nitems % ncols != 0:
                nrows += 1
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            for nplot in nplots:
                fig = plt.figure(figsize=(16, 8))
                fig.suptitle("WXstations [Precipitation]")
                grid = ImageGrid(fig, 111, aspect=False,
                                 nrows_ncols=(nrows, ncols), label_mode="L",
                                 share_all=True, axes_pad=0.5)
                for (axg, rgid,
                     rgdt, rgcs, rgor) in zip(grid,
                                              [i for i in
                                               rg_precip['station_id']
                                               [nplot[0]:nplot[-1]]],
                                              [i for i in
                                               rg_precip['rain_idt']
                                               [nplot[0]:nplot[-1]]],
                                              [i for i in
                                               rg_precip['rain_cumsum']
                                               [nplot[0]:nplot[-1]]],
                                              [i for i in
                                               rg_precip['rain_ires']
                                               [nplot[0]:nplot[-1]]]):
                    axg.set_title(f'Station: {rgid}')
                    axg.plot(np.array(rgdt).flatten(),
                             np.array(rgor).flatten(),
                             '.-', label='Measured Precipitation')
                    axg.plot(np.array(rgdt).flatten(),
                             np.array(rgcs).flatten(),
                             '--',  label='Cumulative Precipitation')
                    axg.xaxis.set_major_locator(locator)
                    axg.xaxis.set_major_formatter(formatter)
                    axg.set_xlabel('Date and time', fontsize=12)
                    axg.set_ylabel('mm', fontsize=12)
                    axg.grid(True)
                    axg.legend()
                    plt.show()
                nitems = len(rg_precip['station_id'])
        self.ds_precip = rg_precip


# =============================================================================
# QPE
# =============================================================================



# =============================================================================
# calib
# =============================================================================
class rhoHV_Noise_Bias:
    r"""
    A class to correct :math:`\rho_{HV}` using the SNR.

    Attributes
    ----------
        elev_angle : float
            Elevation angle at which the scan was taken, in deg.
        file_name : str
            Name of the file containing radar data.
        scandatetime : datetime
            Date and time of scan.
        site_name : str
            Name of the radar site.
        vars : dict
            corrected :math:`\rho_{HV}` and user-defined radar variables.
    """

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def rhohv_noise_correction(self, rad_georef, rad_params, rad_vars,
                               noise_lvl=None, data2correct=None):
        r"""
        Correct :math:`\rho_{HV}` varying the radar constant.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used for the correction method.
            The default is None.
        noise_lvl : 3-element tuple or list, optional
            Interval of radar constant values. [start, stop, step].
            The default is None.
        data2correct : dict, optional
            Dictionary to update the corrected :math:`\rho_{HV}`.
            The default is None.

        Notes
        -----
        1. Based on the method described in [1]_

        References
        ----------
        .. [1] Ryzhkov, A. V.; Zrnic, D. S. (2019). Radar Polarimetry for
            Weather Observations (1st ed.). Springer International Publishing.
            https://doi.org/10.1007/978-3-030-05093-1
        """
        snrhdb = SNR_Classif.static_signalnoiseratio(
            rad_georef, rad_params, rad_vars, rad_cst=noise_lvl,
            snr_linu=True)
        rhohvc = rad_vars['rhoHV [-]']*(1+(1/snrhdb['snr [linear]']))
        if data2correct is not None:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'rhoHV [-]': rhohvc})
            self.vars = data2cc

    def iterate_radcst(self, rad_georef, rad_params, rad_vars, noise_lvl=None,
                       bins_rho=(0.8, 1.1, 0.005), bins_snr=(5, 30, 0.1),
                       rhohv_theo=(0.93, 1.1), calculations='basic',
                       data2correct=None, plot_method=False):
        r"""
        Compute :math:`\rho_{HV}/SNR` dependencies using the radar constant.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used for the correction method.
            The default is None.
        noise_lvl : 3-element tuple or list, optional
            Interval of radar constant values. [start, stop, step].
            The default is None.
        bins_rho : 3-element tuple or list, optional
            Interval of :math:`\rho_{HV}` values. [start, stop, step].
            The default is (0, 1.1, 0.005).
        bins_snr : 3-element tuple or list, optional
            Interval of SNR values. [start, stop, step].
            The default is (5, 50, 0.1).
        rhohv_theo : 2-element tuple or list, optional
            Interval of theoretical :math:`\rho_{HV}` values.
            [start, stop, step]. The default is (0.93, 1.1).
        calculations : str, optional
            'basic' or 'all'. Defines if the outputs describe the whole method
            or only the final result. The default is 'basic'.
        data2correct : dict, optional
            Dictionary to update the corrected :math:`\rho_{HV}`.
            The default is None.
        plot_method : bool, optional
            Plot the iteratiion correction method. The default is False.

        Notes
        -----
        1. Based on the method described in [1]_

        References
        ----------
        .. [1] Ryzhkov, A. V.; Zrnic, D. S. (2019). Radar Polarimetry for
            Weather Observations (1st ed.). Springer International Publishing.
            https://doi.org/10.1007/978-3-030-05093-1
        """
        from sklearn.metrics import root_mean_squared_error as sklrmse

        if noise_lvl is None:
            noise_lvl1 = (0, 100, 5)
            rng_ite = linspace_step(*noise_lvl1)
            tic = time.time()
            snr = [SNR_Classif.static_signalnoiseratio(rad_georef, rad_params,
                                                       rad_vars, rad_cst=n,
                                                       snr_linu=True)
                   for n in rng_ite]
            rhvc = [rad_vars['rhoHV [-]']*(1+1/i['snr [linear]'])
                    for i in snr]
            snrdb = [i['snr [dB]'] for i in snr]
            hists = [np.histogram2d(i[0].flatten(), i[1].flatten(),
                                    bins=(linspace_step(*bins_snr),
                                          linspace_step(*bins_rho)))
                     for i in zip(snrdb, rhvc)]
            histmax = [np.array([i[2][np.argmax(j)] for j in i[0]])
                       for i in hists]
            histrms = [sklrmse(np.linspace(
                rhohv_theo[0], rhohv_theo[1], len(i)), i) for i in histmax]
            idxminstd = np.argmin(histrms)
            rc_comp = rng_ite[idxminstd]

            res2 = 5
            rng_ite2 = linspace_step(rc_comp-res2, rc_comp+res2, 1)
            snr2 = [SNR_Classif.static_signalnoiseratio(rad_georef, rad_params,
                                                        rad_vars, rad_cst=n,
                                                        snr_linu=True)
                    for n in rng_ite2]
            rhvc2 = [rad_vars['rhoHV [-]']*(1+1/i['snr [linear]'])
                     for i in snr2]
            snrdb2 = [i['snr [dB]'] for i in snr2]
            hists2 = [np.histogram2d(i[0].flatten(), i[1].flatten(),
                                     bins=(linspace_step(*bins_snr),
                                           linspace_step(*bins_rho)))
                      for i in zip(snrdb2, rhvc2)]
            histmax2 = [np.array([i[2][np.argmax(j)] for j in i[0]])
                        for i in hists2]
            histrms2 = [sklrmse(np.linspace(
                rhohv_theo[0], rhohv_theo[1], len(i)), i) for i in histmax2]
            # histrms2 = [sklrmse(np.ones_like(i), i) for i in histmax2]
            # histmae2 = [sklmae(np.ones_like(i), i) for i in histmax2]
            # histstats = [(np.ones_like(i), i) for i in histmax2]
            # idxminstd2 = np.argmin(hmxstd2)
            idxminstd2 = np.argmin(histrms2)
            rc_comp2 = rng_ite2[idxminstd2]

            res3 = 0.5
            rng_ite3 = linspace_step(rc_comp2-res3, rc_comp2+res3, .1)
            snr3 = [SNR_Classif.static_signalnoiseratio(rad_georef, rad_params,
                                                        rad_vars, rad_cst=n,
                                                        snr_linu=True)
                    for n in rng_ite3]
            rhvc3 = [rad_vars['rhoHV [-]']*(1+1/i['snr [linear]'])
                     for i in snr3]
            snrdb3 = [i['snr [dB]'] for i in snr3]
            hists3 = [np.histogram2d(i[0].flatten(), i[1].flatten(),
                                     bins=(linspace_step(*bins_snr),
                                           linspace_step(*bins_rho)))
                      for i in zip(snrdb3, rhvc3)]
            histmax3 = [np.array([i[2][np.argmax(j)] for j in i[0]])
                        for i in hists3]
            # hmxstd3 = [i.std() for i in histmax3]
            histrms3 = [sklrmse(np.linspace(
                rhohv_theo[0], rhohv_theo[1], len(i)), i) for i in histmax3]
            # histrms3 = [sklrmse(np.ones_like(i), i) for i in histmax3]
            # histmae3 = [sklmae(np.ones_like(i), i) for i in histmax3]
            # idxminstd3 = np.argmin(hmxstd3)
            idxminstd3 = np.argmin(histrms3)
            rc_comp3 = rng_ite3[idxminstd3]

            res4 = 0.1
            rng_ite4 = linspace_step(rc_comp3-res4, rc_comp3+res4, .01)
            snr4 = [SNR_Classif.static_signalnoiseratio(rad_georef, rad_params,
                                                        rad_vars, rad_cst=n,
                                                        snr_linu=True)
                    for n in rng_ite4]
            rhvc4 = [rad_vars['rhoHV [-]']*(1+1/i['snr [linear]'])
                     for i in snr4]
            snrdb4 = [i['snr [dB]'] for i in snr4]
            hists4 = [list(np.histogram2d(i[0].flatten(), i[1].flatten(),
                                          bins=(linspace_step(*bins_snr),
                                                linspace_step(*bins_rho))))
                      for i in zip(snrdb4, rhvc4)]
            histmax4 = [np.array([i[2][np.argmax(j)] for j in i[0]])
                        for i in hists4]
            # hmxstd4 = [i.std() for i in histmax4]
            histrms4 = [sklrmse(np.linspace(
                rhohv_theo[0], rhohv_theo[1], len(i)), i) for i in histmax4]
            # histrms4 = [sklrmse(np.ones_like(i), i) for i in histmax4]
            # histmae4 = [sklmae(np.ones_like(i), i) for i in histmax4]
            # idxminstd4 = np.argmin(hmxstd4)
            idxminstd4 = np.argmin(histrms4)
            rc_comp4 = rng_ite4[idxminstd4]
            toc = time.time()
            if print_rtimes:
                print(f"SNR iteration running time: {toc-tic:.3f} sec.")
        else:
            if len(noise_lvl) != 3:
                rng_ite4 = linspace_step(*noise_lvl)
            else:
                rng_ite4 = linspace_step(
                    noise_lvl[0], noise_lvl[1]+noise_lvl[2], noise_lvl[2])
            tic = time.time()
            snr4 = [SNR_Classif.static_signalnoiseratio(rad_georef, rad_params,
                                                        rad_vars, rad_cst=n,
                                                        snr_linu=True)
                    for n in rng_ite4]
            rhvc4 = [rad_vars['rhoHV [-]']*(1+1/i['snr [linear]'])
                     for i in snr4]
            snrdb4 = [i['snr [dB]'] for i in snr4]
            hists4 = [list(np.histogram2d(i[0].flatten(), i[1].flatten(),
                                          bins=(linspace_step(*bins_snr),
                                                linspace_step(*bins_rho))))
                      for i in zip(snrdb4, rhvc4)]
            histmax4 = [np.array([i[2][np.argmax(j)] for j in i[0]])
                        for i in hists4]

            # hmxstd4 = [i.std() for i in histmax4]
            histrms4 = [sklrmse(np.linspace(
                rhohv_theo[0], rhohv_theo[1], len(i)), i) for i in histmax4]
            # histrms4 = [sklrmse(np.ones_like(i), i) for i in histmax4]
            # histmae4 = [sklmae(np.ones_like(i), i) for i in histmax4]
            idxminstd4 = np.nanargmin(histrms4)
            rc_comp4 = rng_ite4[idxminstd4]
            toc = time.time()
            if print_rtimes:
                print(f"SNR iteration running time: {toc-tic:.3f} sec.")

        v = {}
        rhov_corr = np.where((rhvc4[idxminstd4] > rhohv_theo[1]),
                             rad_vars['rhoHV [-]'], rhvc4[idxminstd4])
        # rhov_corr = np.where((rhvc4[idxminstd4] > rhohv_theo[1]),
        #                      rhohv_theo[1], rhvc4[idxminstd4])
        v['rhoHV [-]'] = rhov_corr
        rhvc4[idxminstd4] = rhov_corr
        if calculations == 'basic':
            rhohv_corr = {'hist': hists4[idxminstd4],
                          'histmax': histmax4[idxminstd4],
                          'Noise level [dB]': rc_comp4,
                          'SNR [dB]': snrdb4[idxminstd4]}
        elif calculations == 'all':
            rhohv_corr = {'hist': hists4, 'histmax': histmax4,
                          'Noise level [dB]': rc_comp4,
                          'SNR [dB]': snrdb4[idxminstd4]}

        self.rhohv_corrs = rhohv_corr
        if data2correct is None:
            self.vars = v
        else:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'rhoHV [-]': rhvc4[idxminstd4]})
            self.vars = data2cc
        if plot_method:
            plot_rhocalibration(hists4, histmax4, idxminstd4, rng_ite4)


def plot_rhocalibration(hists, histmax, idxminstd, rng_ite, fig_size=None):
    """
    Plot the histograms used to compute the rhoHV correction.

    Parameters
    ----------
    hists : TYPE
        DESCRIPTION.
    histmax : TYPE
        DESCRIPTION.
    idxminstd : TYPE
        DESCRIPTION.
    rng_ite : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nr = 1
    nc = int(len(hists))
    if len(hists) > 3:
        nc = 3
        nr = len(hists) // nc + (len(hists) % nc > 0)

    if fig_size is None:
        fig_size = (16, 9)
    fig, axes = plt.subplots(sharex=True, sharey=True, nrows=nr, ncols=nc,
                             figsize=fig_size, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        if i < len(hists):
            if i == idxminstd:
                ax.set_title(f'{rng_ite[i]:.3f}', c='tab:purple',
                             fontweight='bold')
            else:
                ax.set_title(f'{rng_ite[i]:.3f}', c='tab:gray')
            ax.plot(hists[i][1][1:], histmax[i], color="k", zorder=10)
            ax.axhline(1)
            pcm = ax.pcolormesh(hists[i][1], hists[i][2], hists[i][0].T,
                                norm=mpc.LogNorm(vmin=10**0, vmax=10**1),
                                rasterized=True, cmap='tpylsc_useq_bupkyw')
            ax.tick_params(axis='both', which='major', labelsize=10)
    clb = fig.colorbar(pcm, ax=axes[1:, :], location='right', shrink=0.85)
    clb.ax.set_title('n points')
    fig.supylabel(r'$\rho_{HV}$ [-]')
    fig.supxlabel('SNR [dB]')
