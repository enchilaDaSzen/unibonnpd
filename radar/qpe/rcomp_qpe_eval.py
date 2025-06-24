#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:09:09 2024

@author: dsanchez
"""

import datetime as dt
import os
import pickle
# import copy
from tqdm import tqdm
import numpy as np
from zoneinfo import ZoneInfo
# import towerpy as tp
from towerpy.utils.radutilities import linspace_step
from radar import twpext as tpx
# import wradlib as wrl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from mpl_toolkits.axes_grid1 import ImageGrid
import cartopy.crs as ccrs
from towerpy.datavis import rad_display
from radar.rparams_dwdxpol import RPRODSLTX

# =============================================================================
# Define working directory, time and list files
# =============================================================================
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24h [NO JXP]
START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24h []
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2018, 12, 2, 0, 0)  # 24 h [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24h [NO JXP]
# # START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 8, 0)  # 16h [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24h [NO BXP]

rcomp = 'rcomp_qpe_dwd'
# rcomp = 'rcomp_qpe_dwdbxp'
# rcomp = 'rcomp_qpe_dwdjxp'
rcomp = 'rcomp_qpe_dwdxpol'
# rcomp = 'rcomp_qpe_xpol'

xlims, ylims = [4.324, 10.953], [48.635, 52.754]  # DWDXPOL RADCOV
xlims, ylims = [5.85, 11.], [48.55, 52.75]  # DWDXPOL DE

rprods_dp = ['r_adp', 'r_ah', 'r_kdp', 'r_z']
rprods_hbr = ['r_ah_kdp', 'r_kdp_zdr', 'r_z_ah', 'r_z_kdp', 'r_z_zdr']
rprods_opt = ['r_kdpopt', 'r_zopt']
rprods_hyop = ['r_ah_kdpopt', 'r_zopt_ah', 'r_zopt_kdp', 'r_zopt_kdpopt']
rprods_nmd = ['r_aho_kdpo', 'r_kdpo', 'r_zo', 'r_zo_ah', 'r_zo_kdp',
              'r_zo_zdr']


# rprods = sorted(rprods_dp[1:] + rprods_hbr + rprods_opt + rprods_hyop
#                 + ['r_zo', 'r_aho_kdpo'])
rprods = sorted(rprods_dp[1:] + rprods_hbr + rprods_opt + rprods_hyop
                + ['r_zo', 'r_kdpo'])
# rprods = sorted(rprods_dp[1:] + rprods_hbr)

SAVE_FIGS = True
SAVE_DATA = False
PLOT_METHODS = True

RES_DIR = LWDIR + f"pd_rdres/qpe_{START_TIME.strftime('%Y%m%d')}/{rcomp}/"

if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
    RPRODSLTX['r_aho_kdpo'] = '$R(A_H, K_{DP})[OV]$'
    RPRODSLTX['r_kdpo'] = '$R(K_{DP})[OV]$'
    RPRODSLTX['r_zo'] = '$R(Z_{H})[OA]$'
# %%
# =============================================================================
# Read in Radar QPE
# =============================================================================
qpe_amlb = False
if qpe_amlb:
    appxf = '_amlb'
else:
    appxf = ''

RQPEH_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}/"
             + f"{rcomp}/hourly{appxf}/")
RQPEH_FILES = [RQPEH_DIR + i for i in sorted(os.listdir(RQPEH_DIR))
               # if i.endswith('rhqpe.tpy')
               if 'rhqpe' in i
               ]

RQPE_GRID = [RQPEH_DIR + i for i in sorted(os.listdir(RQPEH_DIR))
             if i.endswith('mgrid.tpy')]
RQPE_PARAMS = [RQPEH_DIR + i for i in sorted(os.listdir(RQPEH_DIR))
               if i.endswith('params.tpy')]


def qpegeoref_reader(RQPE_GRID):
    """Read in the georeference data for the qpe."""
    for qpef in RQPE_GRID:
        with open(qpef, 'rb') as fpkl:
            gridqpe = pickle.load(fpkl)
        return gridqpe


def qpeparams_reader(RQPE_PARAMS):
    """Read in the params data for the qpe."""
    for qpef in RQPE_PARAMS:
        with open(qpef, 'rb') as fpkl:
            qpe_pars = pickle.load(fpkl)
        return qpe_pars


# %%
# =============================================================================
# Read hourly acummulations
# =============================================================================
qpe_georef = qpegeoref_reader(RQPE_GRID)
qpe_params = qpeparams_reader(RQPE_PARAMS)

# Read hourly acummulations
resqpedf = {}
for rp in rprods:
    # rp = rprods[1]
    resqpeh = []
    for cnt, h_acc in enumerate(
            tqdm(RQPEH_FILES, desc=f'Reading hourly accumulations [{rp}]')):
        # print(h_acc)
        if h_acc.endswith(f'rhqpe_{rp}.tpy'):
            with open(h_acc, 'rb') as fpkl:
                resqpe = pickle.load(fpkl)
                resqpe = [*resqpe.values()][0]
                if len(resqpe['elev_ang [deg]']) > 1:
                    resqpeh.append(resqpe)
    # Compute daily accumulations
    for cnt, h_acc in enumerate(
            tqdm(resqpeh, desc=f'Computing daily accumulations [{rp}]')):
        # print(cnt)
        if cnt == 0:
            resqped = {k: v for k, v in h_acc.items()
                       if k.startswith('r_')}
        # elif cnt < 23:
        else:
            for k in resqped.keys():
                resqped[k] = np.nansum((resqped[k], h_acc[k]), axis=0)
        resqpedf[rp] = resqped[rp]

# %%
resqpe_accd = {k: v for k, v in resqpedf.items()
               if k.startswith('r_')}
resqpe_accd_params = {}
resqpe_accd_params['datetime'] = resqpe['datetime'].replace(
    tzinfo=ZoneInfo('Europe/Berlin'))
resqpe_accd_params['elev_ang [deg]'] = resqpe['elev_ang [deg]']

# %%
# =============================================================================
# Read RG data
# =============================================================================
RG_WDIR = EWDIR + 'pd_rdres/dwd_rg/'
DWDRG_MDFN = (RG_WDIR + 'RR_Stundenwerte_Beschreibung_Stationen2024.csv')
RG_NCDATA = (RG_WDIR + f"nrw_{START_TIME.strftime('%Y%m%d')}_"
             + f"{(START_TIME+dt.timedelta(hours=24)).strftime('%Y%m%d')}"
             + "_1h_1hac/")
# =============================================================================
# Init raingauge object
# =============================================================================
rg_data = tpx.RainGauge(RG_WDIR, nwk_opr='DWD')

# =============================================================================
# Read metadata of all DWD rain gauges (location, records, etc)
# =============================================================================
rg_data.get_dwdstn_mdata(DWDRG_MDFN, plot_methods=False)

# =============================================================================
# Get rg locations within bounding box
# =============================================================================
# bbox_xlims, bbox_ylims = (6, 9.2), (49.35, 52.32)  # XPOL
# bbox_xlims, bbox_ylims = (5.5, 8.5), (49.7, 52.18)  # XPOLF
# bbox_xlims, bbox_ylims = (6, 11.), (48.6, 52.8)  # DWDXPOL
bbox_xlims, bbox_ylims = (6, 10.7), (49, 52.6)  # DWDXPOLF

# bbox_xlims, bbox_ylims = (7.85, 7.9), (49.5, 49.55)  # EDRMV

# corr = [2667, 4741]
corr = []
fid = {'20170724': [567, 4132, 5279, 6197, 3904, 450, 535, 732, 953, 2619,
                    5871, 14076, 14077, 14082, 14088, 14104, 2810, 5103, 5646,
                    3700, 7375, 2486, 4310, 3263, 377, 14054, 2184, 5275, 6260,
                    7495, 3042, 5294, 3527, 4849, 7374, 5433, 13674, 3969],
       '20170725': [1241, 1246, 1766, 2027, 2323, 2703, 2810, 2999, 3844,
                    4288, 4371, 4849, 6197, 7106, 7425, 13669, 13675, 15767,
                    15771, 1039,  1691,  2171,  4763,  5646, 19312, 91, 211,
                    1639, 5416, 130, 240, 2880, 3700, 4480, 6287, 6336, 7104,
                    7227, 19246, 19271, 755, 1093, 1216, 3448, 6244, 7429,
                    3024, 4301, 4308, 4709, 6186, 14012, 14029, 14048, 14139,
                    15198, 15821, 17591, 17592, 19574, 19858, 2486, 3625,
                    191, 378, 505, 1095, 1297, 1411, 1573, 1997, 2152, 2503,
                    2597, 3207, 3348, 3560, 3650, 3836, 4135, 4377, 4978, 5084,
                    5371, 6272, 6347, 7250, 7256, 7259, 7273, 19134, 19142,
                    19158, 19225, 19316, 19448, 1526, 2066, 3911, 357, 7375,
                    4310, 1598],
       '20180516': [1580, 2293, 4301, 4308, 6186, 6329, 14044, 14045, 14046,
                    14063, 15569, 17592, 19303, 19304, 19311, 19859, 755, 5906,
                    978, 5294, 6344, 535, 5871, 1964, 1055, 685, 7187, 7412,
                    240, 4480, 4560, 953, 1170, 91],
       '20180923': [3625, 4490, 5029, 5433, 1255, 1650, 1411, 2597, 4377, 5371,
                    19225, 19448, 505, 1297, 2152, 2925, 4135, 4763, 5084,
                    7273, 19158, 19316, 294, 342, 1304, 3081, 3820, 3913,
                    4790, 4063, 4112, 1573, 2503, 3207, 3560, 3836, 4978, 6347,
                    2600, 2709, 4603, 5149, 6201, 2787, 3733, 3734, 3761, 4719,
                    5711, 5990, 6245, 6259, 7229, 7237, 7490, 7498, 15012,
                    460, 2331, 3545, 4336, 6217, 13778, 5513, 3939, 5989, 6260,
                    7230, 7495, 789, 3257, 211, 389, 390, 7249],
       '20181202': [161, 348, 535, 2167, 2595, 3024, 3969, 4167, 4650, 4709,
                    6242, 6340, 13689, 14011, 14015, 14020, 14029, 14032,
                    14038, 14043, 14049, 14079, 14081, 14139, 15198, 15212,
                    15821, 17587, 17593, 19100, 19861, 732, 953, 6333, 15570,
                    15989, 2999,  4288,  4367,  4371,  4849,  7106,  7374,
                    13675, 1039, 1170, 1691, 3348, 1297, 4763, 1300, 2947,
                    4488, 2027, 1055, 3042, 5294, 3844, 211, 390, 5416, 7249,
                    240, 6336, 7227, 7396, 6186, 14006, 3263, 5433, 6324,
                    15040, 15828, 6344, 6318],

       '20190508': [1580, 1645, 3028, 5099, 5100, 6337],
       '20190511': [],
       '20190720': [342, 5513, 4063, 2968, 3820, 2174, 2175, 1411],
       '20200617': [13674, 3340, 357, 130, 3167, 5294, 15012, 377, 14054, 1650,
                    5029, 2597, 4377, 2925, 3042],
       '20210713': [336, 5297, 1223, 194, 6217, 2331, 731, 7498, 2618, 5335,
                    7490, 4336, 3207, 6347, 3734, 5711, 2787],
       '20210714': [2667, 4741, 6305, 1223, 2562, 6217],
       }
# fid = {k1: corr for k1, v1 in fid.items()}

rg_data.get_stns_box(rg_data.dwd_stn_mdata, bbox_xlims=bbox_xlims,
                     bbox_ylims=bbox_ylims, plot_methods=False,
                     surface=qpe_georef, isrfbins=9, dmax2srfbin=1,
                     del_by_station_id=fid[START_TIME.strftime('%Y%m%d')])

# rg2del = rg_data.stn_bbox
# rg2del['station_id']
# %%
# =============================================================================
# Download DWD rg data
# =============================================================================
# for hour in range(72):
#     start_time = dt.datetime(2021, 7, 13, 0, 0, 0)
#     # print(hour)
#     start_time = start_time + dt.timedelta(hours=hour)
#     print(start_time)
#     stop_time = start_time + dt.timedelta(hours=1)
#     print(stop_time)
#     # start_time = start_time + datetime.timedelta(hours=hour+1)
#     # print(start_time)
#     for station_id in rg_data.stn_near_rad['stations_id']:
#         rg_data.get_dwdstn_nc(station_id, start_time, stop_time,
#                               dir_ncdf=rg_ncdata)

# =============================================================================
# Read DWD rg data
# =============================================================================
rg_data.get_rgdata(resqpe_accd_params['datetime'], ds_ncdir=RG_NCDATA,
                   drop_nan=True, drop_thrb=0.1, ds2read=rg_data.stn_bbox,
                   plot_methods=False, ds_tres=dt.timedelta(hours=1),
                   rprod_fltr=resqpe_accd['r_zopt'], rprod_thr=0.1,
                   # rprod_fltr=resqpe_accd['r_z'], rprod_thr=0.5,
                   dt_bkwd=dt.timedelta(hours=25),
                   ds_accum=dt.timedelta(hours=25))
# %%
# rdatah = {'beam_height [km]': rqpe_acc.georef['beam_height [km]']}
rg_acprecip = {'grid_wgs84x': rg_data.ds_precip['longitude [dd]'],
               'grid_wgs84y': rg_data.ds_precip['latitude [dd]'],
               'Rainfall [mm]': rg_data.ds_precip['rain_sum'].flatten(),
               'altitude [m]': rg_data.ds_precip['altitude [m]'].flatten()}
# %%

# fig, ax = plt.subplots()

# ax.boxplot(rg_acprecip['Rainfall [mm]'], notch=True)

# plt.show()

# %%
rprods_plots = rprods

# rprods_plots = ['r_kdp', 'r_kdpo', 'r_kdpopt', 'r_z', 'r_zo', 'r_zopt']
# SAVE_FIGS = False
if PLOT_METHODS:
    for rprodk in sorted(rprods_plots):
        if rprodk in RPRODSLTX:
            rprodkltx = RPRODSLTX.get(rprodk)
        else:
            rprodkltx = rprodk
        # axg.set_title(f'{rprodkltx}')
        bnd = {}
        bnd['[mm]'] = np.array((0.1, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35,
                                40, 45, 50, 55, 60, 70, 80, 90, 100, 115, 130,
                                145, 160, 175, 200))
        unorm = {}
        unorm['[mm]'] = mpc.BoundaryNorm(
            bnd['[mm]'], mpl.colormaps['tpylsc_rad_rainrt'].N, extend='max')
        rqpe_acc = {}
        rqpe_acc['Rainfall [mm]'] = resqpedf[rprodk]
        rad_display.plot_ppi(
            qpe_georef, resqpe_accd_params, rqpe_acc, fig_size=(10.5, 7),
            cpy_feats={'status': True}, xlims=xlims, ylims=ylims,
            points2plot=rg_acprecip, ptsvar2plot='Rainfall [mm]',
            data_proj=ccrs.PlateCarree(), proj_suffix='wgs84',
            fig_title=(
                f'Radar Composite [{rprodkltx}]: '
                + f"{resqpe_accd_params['datetime']:%Y-%m-%d}"),
            # ucmap='jet',
            # unorm=unorm
            font_sizes='large'
            )

        if SAVE_FIGS:
            fname = (f"{START_TIME.strftime('%Y%m%d')}"
                     + f'_{rcomp}_accum_24h_'
                     + f"{rprodk.replace('_', '')}{appxf}L.png")
            plt.savefig(RES_DIR+fname, format='png', dpi=300)
            plt.close()

# %%
cmaph = mpl.colormaps['gist_earth_r']
cmaph = mpl.colormaps['Spectral_r']
# lpv = {'Altitude [m]':
#        [round(np.nanmin(rg_data.ds_precip['altitude [m]']), 2),
#         round(np.nanmax(rg_data.ds_precip['altitude [m]']), -2), 25]}
lpv = {'Altitude [m]': [0, 750, 11]}
bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
       for key, value in lpv.items()}
dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
    value, cmaph.N, extend='max') for key, value in bnd.items()}

eval_rqp = {k: np.array(
    [np.nanmean(i)
     for i in restimator.flatten()[rg_data.ds_precip['kd_rbin_idx']]])
    for k, restimator in resqpe_accd.items()}

eval_rng = {k: rg_data.ds_precip['rain_sum'].flatten()
            for k in resqpe_accd.keys()}

for k, v in eval_rng.items():
    v[np.isnan(eval_rqp[k])] = np.nan
    v[eval_rqp[k] <= 0.1] = np.nan
    # v[v <= 0.1] = np.nan

qpe_stats = {k: tpx.mstats(v, eval_rng[k], rmse_norm='mean')
             for k, v in eval_rqp.items()}

maxplt = 18
nitems = len(eval_rqp)
nplots = [[i*maxplt, (1+i)*maxplt]
          for i in range(int(np.ceil(nitems/maxplt)))]
nplots[-1][-1] = nitems
if nitems > maxplt:
    nitems = maxplt
nrows = int(nitems**0.5)
ncols = nitems // nrows
# Number of rows, add one if necessary
if nitems % ncols != 0:
    nrows += 1
# locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
# formatter = mdates.ConciseDateFormatter(locator)
if PLOT_METHODS:
    for nplot in nplots:
        fig = plt.figure(figsize=(19.2, 11))
        fig.suptitle('Daily accumulated radar QPE vs Rain-gauge measured'
                     f" rain totals [{START_TIME.strftime('%Y-%m-%d')}] \n",
                     size=16)
        grid = ImageGrid(fig, 111, aspect=False,
                         nrows_ncols=(nrows, ncols), label_mode='L',
                         share_all=True, axes_pad=0.5,  cbar_location="right",
                         cbar_mode="single", cbar_size="4%", cbar_pad=0.5)
        for (axg, rprodk) in zip(grid, [k for k in sorted(eval_rqp.keys())]):
            if rprodk in RPRODSLTX:
                rprodkltx = RPRODSLTX.get(rprodk)
            else:
                rprodkltx = rprodk
            axg.set_title(f'{rprodkltx}', size=14)
            f1 = axg.scatter(eval_rng[rprodk], eval_rqp[rprodk], marker='o',
                             c=[rg_acprecip['altitude [m]']], edgecolors='k',
                             cmap=cmaph, norm=dnorm['nAltitude [m]'])
            f2 = axg.scatter(
                0, 0, marker='',
                label=(
                    f"n={qpe_stats[rprodk]['N']}"
                    + f"\nr={qpe_stats[rprodk]['R_Pearson [-]'][0,1]:.2f}"
                    + f"\nMAE={qpe_stats[rprodk]['MAE']:2.2f}"
                    + f"\nRMSE={qpe_stats[rprodk]['RMSE']:2.2f}"
                    # + f"\nNRMSE [%]={qpe_stats[rprodk]['NRMSE [%]']:2.2f}"
                    # + f"\nNMB [%]={qpe_stats[rprodk]['NMB [%]']:2.2f}"
                    ))
            axg.axline((1, 1), slope=1, c='gray', ls='--')
            axg.set_xlabel('Rain-gauge rainfall [mm]', fontsize=14)
            axg.set_ylabel('Radar rainfall [mm]', fontsize=14)
            if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
                axg.set_xlim([0, 85])
                axg.set_ylim([0, 85])
            else:
                axg.set_xlim([0, 180])
                axg.set_ylim([0, 180])
            axg.grid(True)
            axg.legend(loc=2, fontsize=12, handlelength=0, handletextpad=0,
                       fancybox=True)
            # axg.xaxis.set_tick_params(labelsize=12)
            axg.tick_params(axis='both', which='major', labelsize=12)
            plt.show()
        # nitems = len(eval_rqp)
        axg.cax.colorbar(f1)
        axg.cax.tick_params(direction='in', which='both', labelsize=14)
        # axg.cax.toggle_label(True)
        axg.cax.set_title('altitude [m]', fontsize=14)
        plt.tight_layout()
    if SAVE_FIGS:
        fname = (f"{START_TIME.strftime('%Y%m%d')}"
                 + f"_{rcomp}_accum_24h{appxf}.png")
        plt.savefig(RES_DIR + fname, format='png')

# %%
theta = np.linspace(0.0, 2 * np.pi, len(qpe_stats), endpoint=False)

stat2plot = 'MAE'
stat2plot = 'RMSE'
stat2plot = 'NRMSE [%]'
stat2plot = 'NMB [%]'
stat2plot = 'R_Pearson [-]'

if stat2plot == 'R_Pearson [-]':
    colors = plt.get_cmap('Spectral')
elif stat2plot == 'NMB [%]':
    # colors = plt.get_cmap('tpylsc_div_dbu_w_rd')
    colors = plt.get_cmap('tpylsc_div_dbu_rd')
    # colors = plt.get_cmap('tpylsc_div_yw_gy_bu')
    # colors = plt.get_cmap('berlin')
else:
    colors = plt.get_cmap('Spectral_r')


bnd = {}
# bnd['MAE'] = np.linspace(0, 20, 11)
# bnd['RMSE'] = np.linspace(0, 20, 11)
# bnd['NRMSE [%]'] = np.linspace(35, 65, 16)
# bnd['NRMSE [%]'] = np.linspace(-10, 100, 23)
# bnd['NMB [%]'] = np.linspace(-50, 50, 11)
# bnd['R_Pearson [-]'] = np.linspace(0.8, 1, 11)
bnd['MAE'] = linspace_step(0, 20, 1)
bnd['RMSE'] = linspace_step(0, 20, 1)
bnd['NRMSE [%]'] = linspace_step(30, 80, 5)
bnd['NMB [%]'] = linspace_step(-50, 50, 10)
bnd['R_Pearson [-]'] = linspace_step(0.7, 1, 0.02)
dnorm = {}
dnorm['MAE'] = mpc.BoundaryNorm(bnd['MAE'], colors.N, extend='max')
dnorm['RMSE'] = mpc.BoundaryNorm(bnd['RMSE'], colors.N, extend='max')
dnorm['NRMSE [%]'] = mpc.BoundaryNorm(bnd['NRMSE [%]'], colors.N,
                                      extend='max')
dnorm['NMB [%]'] = mpc.BoundaryNorm(bnd['NMB [%]'], colors.N, extend='both')
dnorm['R_Pearson [-]'] = mpc.BoundaryNorm(bnd['R_Pearson [-]'],
                                          colors.N, extend='min')


def colored_bar(left, height, z=None, width=0.5, bottom=0, ax=None, **kwargs):
    import itertools
    # import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    if ax is None:
        ax = plt.gca()
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    rects = []
    for x, y, h, w in zip(left, bottom, height, width):
        rects.append(Rectangle((x, y), w, h))
    coll = PatchCollection(rects, array=z, **kwargs)
    ax.add_collection(coll)
    ax.autoscale(enable=False, axis='y', tight=True)
    return coll


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='polar')
if stat2plot == 'R_Pearson [-]':
    coll = colored_bar(theta,
                       [st[stat2plot][0][1] for k, st in qpe_stats.items()],
                       z=[st[stat2plot][0][1] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       cmap=colors, norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend=dnorm[stat2plot].extend, pad=.1)
elif stat2plot == 'MAE' or stat2plot == 'RMSE':
    coll = colored_bar(theta, [st[stat2plot] for k, st in qpe_stats.items()],
                       z=[st[stat2plot] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       cmap=colors, norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend=dnorm[stat2plot].extend, pad=.1)
else:
    coll = colored_bar(theta, [st[stat2plot] for k, st in qpe_stats.items()],
                       z=[st[stat2plot] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       cmap=colors, norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend=dnorm[stat2plot].extend, pad=.1)
    if stat2plot == 'NMB [%]':
        plt.polar(np.arange(0, (2 * np.pi), 0.01),
                  np.zeros_like(np.arange(0, (2 * np.pi), 0.01)), 'k--')
cb.ax.set_title(stat2plot, size=18)
cb.ax.tick_params(labelsize=14)
ax.grid(color='gray', linestyle=':')
ax.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticklabels([])
if stat2plot == 'R_Pearson [-]':
    plt.rgrids(np.arange(.10, 1.01, .1), angle=0, size=14, fmt='%.1f', c='k')
    y = 1.183
elif stat2plot == 'NRMSE [%]':
    plt.rgrids(linspace_step(0, bnd['NRMSE [%]'].max(), 10), angle=0, size=14,
               fmt='%.0f', c='k')
    y = bnd['NRMSE [%]'].max()*1.183
    # y = bnd['NRMSE [%]'].max()*1.2
elif stat2plot == 'NMB [%]':
    plt.rgrids(np.arange(-50, 60, 10), angle=0, size=14, fmt='%.1f', c='k')
    y = 68.3
else:
    plt.rgrids(np.arange(0, 18., 2), angle=0, size=14, fmt='%.0f', c='k')
    y = 18.94
# plt.rgrids(np.arange(0, 16., 5), angle=90, size=10, fmt='%.2f')
for c1, v1 in enumerate(qpe_stats):
    x = (((np.deg2rad(360/len(qpe_stats)))/2)
         + ((np.deg2rad(360/len(qpe_stats)))*c1))
    ax.text(x, y, f'{RPRODSLTX.get(v1)}', ha='center', va='center',
            size=14)
# pos = ax.get_rlabel_position()
# ax.set_rlabel_position(pos+157.5)
plt.tight_layout()
# ax.set_title(f'bar length:''\n''RMSE [km]', fontsize=28, x=0, y=-.1)
# ax.set_title(f'bar length: \n {stat2plot}', fontsize=12, x=0, y=-.08)
plt.show()

if SAVE_FIGS:
    if '[' in stat2plot:
        fnst = stat2plot[:stat2plot.find('[')-1]
    else:
        fnst = stat2plot
    fndata = f"{START_TIME.strftime('%Y%m%d')}_{rcomp}_stats{appxf}{fnst}.png"
    plt.savefig(RES_DIR + fndata, format='png')


# %%


frstats = {'dt': START_TIME.strftime('%Y%m%d'),
           'eval_rng': eval_rng, 'eval_rqp': eval_rqp,
           'altitude [m]': rg_acprecip['altitude [m]']}
if SAVE_DATA:
    fndata = f"{START_TIME.strftime('%Y%m%d')}_{rcomp}_stats{appxf}.tpy"
    with open(RES_DIR + fndata, 'wb') as f:
        pickle.dump(frstats, f, pickle.HIGHEST_PROTOCOL)
