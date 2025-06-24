#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:22:51 2022

@author: dsanchez
"""

import datetime as dt
import pickle
import numpy as np
# from tqdm import tqdm
import towerpy as tp
from radar import twpext as tpx
import os
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

# =============================================================================
# Define working directory and list files
# =============================================================================
# Boxpol Juxpol, Essen, Flechtdorf, Neuheilenbach, Offenthal, Hannover
RADAR_SITE = 'Neuheilenbach'
PTYPE = 'qvps'

fullqc = True
read_mlcal = True

PLOT_METHODS = False
SAVE_FIGS = False


# DTWORK = dt.datetime(2017, 7, 19, 0, 0)
DTWORK = dt.datetime(2017, 7, 24, 0, 0)
# DTWORK = dt.datetime(2017, 7, 25, 0, 0)
# DTWORK = dt.datetime(2018, 5, 16, 0, 0)
# DTWORK = dt.datetime(2018, 9, 23, 0, 0)
DTWORK = dt.datetime(2018, 12, 2, 0, 0)
# DTWORK = dt.datetime(2019, 5, 8, 0, 0)
# DTWORK = dt.datetime(2019, 5, 11, 0, 0)
# DTWORK = dt.datetime(2019, 5, 20, 0, 0)
# DTWORK = dt.datetime(2019, 7, 20, 0, 0)
# DTWORK = dt.datetime(2020, 6, 13, 0, 0)  # NO BXP
# DTWORK = dt.datetime(2020, 6, 17, 0, 0)  # NO BXP
# DTWORK = dt.datetime(2021, 2, 6, 0, 0)  # 24 hr
# DTWORK = dt.datetime(2021, 7, 13, 0, 0)
DTWORK = dt.datetime(2021, 7, 14, 0, 0)
# DTWORK = dt.datetime(2019, 5, 10, 0, 0)
# DTWORK = dt.datetime(2019, 5, 11, 0, 0)
# DTWORK = dt.datetime(2021, 6, 20, 0, 0)  # 24 hr


LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

extend_mlyr = False
if extend_mlyr:
    appx = '_extmlyr'
else:
    appx = ''
# appx = '_wrongrhvc'
if 'xpol' in RADAR_SITE:
    if read_mlcal:
        WDIR = (EWDIR + f'pd_rdres/{PTYPE}/xpol/qc/')
    else:
        WDIR = (EWDIR + f'pd_rdres/{PTYPE}/xpol/')
    if fullqc:
        WDIR = (EWDIR + f'pd_rdres/{PTYPE}/xpol/fqc/')
else:
    if read_mlcal:
        WDIR = (EWDIR + f'pd_rdres/{PTYPE}/dwd/qc/')
    else:
        WDIR = (EWDIR + f'pd_rdres/{PTYPE}/dwd/')
    if fullqc:
        WDIR = (EWDIR + f'pd_rdres/{PTYPE}/dwd/fqc/')

# if fullqc:
PPFILES = [WDIR+i for i in sorted(os.listdir(WDIR))
           if i.endswith(f'{PTYPE}.tpy') and RADAR_SITE in i
           and i.startswith(f"{DTWORK.strftime('%Y%m%d')}")]
# else:
#     PPFILES = [WDIR+i for i in sorted(os.listdir(WDIR))
#                if i.endswith(f'_{PTYPE}.tpy') and RADAR_SITE in i
#                and f"{DTWORK.strftime('%Y%m%d')}" in i]
RES_DIR = LWDIR + f"pd_rdres/qvps_d4calib{appx}/{DTWORK.strftime('%Y%m%d')}/"
if read_mlcal:
    RCFILES = [RES_DIR+i for i in sorted(os.listdir(RES_DIR))
               if i.endswith('qvps.tpy') and RADAR_SITE in i]



# %%
# =============================================================================
# Read radar profiles
# =============================================================================
with open(PPFILES[0], 'rb') as f:
    rprofs = pickle.load(f)

# rprofs = rprofs[288:]
if read_mlcal:
    with open(RCFILES[0], 'rb') as f:
        rprfc = pickle.load(f)
else:
    rprfc = []

# %%
# =============================================================================
# ZH Offset adjustment
# =============================================================================
zh_oc = False
if zh_oc:
    RSITESH = {'Boxpol': 3.5, 'Juxpol': 5, 'Essen': 0,
               'Flechtdorf': 0, 'Neuheilenbach': 0,
               'Offenthal': 0}
    # Adjust zh offset
    for rp in rprofs:
        rp.qvps['ZH [dBZ]'] += 3.5

# =============================================================================
# ZDR bias adjustment
# =============================================================================
zdr_oc = False
if read_mlcal:
    zdro = np.array([i.zdr_offset for i in rprfc['zdrO']])
    print(f'nan_elm = {np.count_nonzero(np.isnan(zdro))}')
    print(f'zero_elm = {np.count_nonzero(zdro==0)}')

if zdr_oc:
    for cnt, rp in enumerate(rprofs):
        rp.qvps['ZDR [dB]'] -= zdro[cnt]

# =============================================================================
# PhiDP bias adjustment
# =============================================================================
phidp_oc = False
if read_mlcal:
    phidpo = np.array([i.phidp_offset for i in rprfc['phidpO']])
if phidp_oc:
    for cnt, rp in enumerate(rprofs):
        rp.qvps['PhiDP [deg]'] -= phidpo[cnt]

# =============================================================================
# Adjust relative height
# =============================================================================
adjh = True
if adjh:
    RSITESH = {'Boxpol': 99.50, 'Juxpol': 310.00, 'Essen': 185.11,
               'Flechtdorf': 627.88, 'Neuheilenbach': 585.85,
               'Offenthal': 245.80, 'Hannover': 97.66}
    # Add rheight to mlyrs to work with hAMSL
    if read_mlcal:
        for ml in rprfc['mlyr']:
            ml.ml_top = ml.ml_top + RSITESH[RADAR_SITE]/1000
            ml.ml_bottom = ml.ml_bottom + RSITESH[RADAR_SITE]/1000
            ml.thickness = ml.ml_top - ml.ml_bottom
            # ml.ml_bottom += RSITESH[RADAR_SITE]/1000
        # Add rheight to profs to work with hAMSL
    for pr in rprofs:
        pr.georef['profiles_height [km]'] += RSITESH[RADAR_SITE]/1000

# prof_pcp_type = np.array([i.pcp_type for i in rprofs])
# %%
if read_mlcal:
    ml_top = [i.ml_top for i in rprfc["mlyr"]]
    print(f'ML_TOP: {np.nanmean(ml_top):.2f}')
    ml_btm = [i.ml_bottom for i in rprfc["mlyr"]]
    print(f'ML_BTM: {np.nanmean(ml_btm):.2f}')
    ml_thk = [i.ml_thickness for i in rprfc["mlyr"]]
    print(f'ML_THK: {np.nanmean(ml_thk):.2f}')

if PLOT_METHODS:
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=(True))
    axs = ax[0]
    axs.set_title('Offset variation using the QVPs method')
    axs.plot([i.scandatetime for i in rprofs],
             np.array([i.zdr_offset for i in rprfc['zdrO']]),
             marker='o', ms=5, mfc='None', label='QVPs data')
    axs.grid(axis='y')
    axs.tick_params(axis='both', labelsize=10)
    axs.set_ylabel(r'$Z_{DR}$ [dB]', fontsize=10)
    axs = ax[1]
    axs.plot([i.scandatetime for i in rprofs],
             np.array([i.phidp_offset for i in rprfc['phidpO']]),
             marker='o', ms=5, mfc='None', label='QVPs data')
    axs.grid(axis='y')
    axs.tick_params(axis='both', labelsize=10)
    axs.set_ylabel(r'$\Phi_{DP}$ [deg]', fontsize=10)
    axs.set_xlabel('Datetime', fontsize=10)
    # plt.xlim([dt.datetime(2018, 1, 1, 0, 0), dt.datetime(2019, 1, 1, 0, 0)])
    # plt.ylim([-0.4, 0])
    plt.tight_layout()
# %%
# for robj in rprofs:
#     robj.qvps['PhiDP [deg]'] *= -1
# x = rprofs[125].qvps['PhiDP [deg]']
# b = np.all(x[:-1] < x[1:])
tz = 'Europe/Berlin'
htixlim = [dt.datetime(2017, 7, 25, 0, 0).replace(tzinfo=ZoneInfo(tz)),
           dt.datetime(2017, 7, 25, 23, 59).replace(tzinfo=ZoneInfo(tz))]
# htixlim = [dt.datetime(2018, 5, 16, 0, 1).replace(tzinfo=ZoneInfo(tz)),
#            dt.datetime(2018, 5, 16, 23, 59).replace(tzinfo=ZoneInfo(tz))]
# htixlim = [dt.datetime(2019, 5, 8, 0, 0).replace(tzinfo=ZoneInfo(tz)),
#            dt.datetime(2019, 5, 8, 23, 59).replace(tzinfo=ZoneInfo(tz))]
# htixlim = [dt.datetime(2019, 5, 10, 22, 0).replace(tzinfo=ZoneInfo(tz)),
#            dt.datetime(2019, 5, 11, 12, 59).replace(tzinfo=ZoneInfo(tz))]
# htixlim = [dt.datetime(2020, 6, 16, 0, 0).replace(tzinfo=ZoneInfo(tz)),
#               dt.datetime(2020, 6, 16, 23, 59).replace(tzinfo=ZoneInfo(tz))]
# htixlim = [dt.datetime(2021, 7, 14, 0, 0).replace(tzinfo=ZoneInfo(tz)),
#            dt.datetime(2021, 7, 14, 23, 59).replace(tzinfo=ZoneInfo(tz))]
htixlim = None
htixlim = [
    DTWORK.replace(tzinfo=ZoneInfo(tz)),
    (DTWORK + dt.timedelta(seconds=86399)).replace(tzinfo=ZoneInfo(tz))]

if fullqc:
    for rp1 in rprofs:
        rp1.qvps['ZH- [dBZ]'] = rp1.qvps['ZH+ [dBZ]'] - rp1.qvps['ZH [dBZ]']
    
v2p = 'PhiDP [deg]'
v2p = 'KDP [deg/km]'
# v2p = 'AH [dB/km]'
# v2p = 'ZH+ [dBZ]'
# v2p = 'ZH [dBZ]'
# v2p = 'ZDR [dB]'
# v2p = 'rhoHV [-]'
# v2p = 'bin_class [0-5]'
# v2p = 'prof_type [0-6]'

pbins_class = {'no_rain': 0.5, 'light_rain': 1.5, 'modrt_rain': 2.5,
               'heavy_rain': 3.5, 'mixed_pcpn': 4.5, 'solid_pcpn': 5.5}
prof_type = {'NR': 0.5, 'LR [STR]': 1.5, 'MR [STR]': 2.5, 'HR [STR]': 3.5,
             'LR [CNV]': 4.5, 'MR [CNV]': 5.5, 'HR [CNV]': 6.5}


if v2p == 'bin_class [0-5]':
    ptype = 'pseudo'
    ucmap = 'tpylsc_rad_model'
    cbticks = pbins_class
    contourl = 'ZH [dBZ]'
elif v2p == 'prof_type [0-6]':
    ptype = 'pseudo'
    ucmap = 'coolwarm'
    ucmap = 'tpylsc_div_dbu_rd_r'
    ucmap = 'terrain'
    # ucmap = 'cividis'
    cbticks = prof_type
    contourl = 'ZH [dBZ]'
elif v2p == 'ZH+ [dBZ]' or v2p == 'ZHa [dBZ]':
    ucmap = 'tpylsc_rad_ref'
elif v2p == 'ZH- [dBZ]':
    ucmap = 'tpylsc_div_rd_w_k'
# elif v2p == 'KDP [deg/km]' or v2p == 'AH [dB/km]':
    # contourl = 'ZH [dBZ]'
    # contourl = None
    # ptype = 'fcontour'
    # ptype = 'pseudo'
else:
    ptype = 'fcontour'
    # ptype = 'pseudo'
    ucmap = None
    cbticks = None
    contourl = None

# import matplotlib.colors as mpc
# import matplotlib as mpl

# colors_prabhakar = np.array([
#                              [0.00, 0.70, 0.93],
#                              [0.00, 0.00, 1.00],
#                              [0.50, 1.00, 0.00],
#                              [0.40, 0.80, 0.00],
#                              [0.27, 0.55, 0.00],
#                              [1.00, 1.00, 0.00],
#                              [0.80, 0.80, 0.00],
#                              [1.00, 0.65, 0.00],
#                              [1.00, 0.27, 0.00],
#                              [0.80, 0.22, 0.00],
#                              [0.55, 0.15, 0.00],
#                              [1.00, 0.00, 1.00],
#                              [0.58, 0.44, 0.86]])

# cmap_prabhakar = mpl.colors.ListedColormap(colors_prabhakar)
# bnd = {}
# bnd['bKDP [deg/km]'] = np.array([-0.5, -0.25, 0., 0.05, 0.1, 0.2, 0.3, 0.45,
#                                 0.6, 0.8, 1, 2 ,3])
# bnd['bZDR [dB]'] = np.array([-1., -0.5, 0., 0.1, 0.2, 0.3, 0.4, 0.5,
#                                 0.6, 0.8, 1, 2 ,3])
# unorm = {}
# unorm['nKDP [deg/km]'] = mpc.BoundaryNorm(
#     bnd['bKDP [deg/km]'], cmap_prabhakar.N)
# unorm['nZDR [dB]'] = mpc.BoundaryNorm(
#     bnd['bZDR [dB]'], cmap_prabhakar.N, extend='max')

radb = tp.datavis.rad_interactive.hti_base(
    rprofs, mlyrs=(rprfc['mlyr'] if read_mlcal else None),
    var2plot=v2p, stats=None,  # stats='std_dev',
    vars_bounds={'bin_class [0-5]': (0, 6, 7),
                 'prof_type [0-6]': (0, 7, 8),
                 'PhiDP [deg]': [0, 90, 10],
                 'KDP [deg/km]': [-0.4, 1.2, 17],  # [-0.20, 0.6, 17],
                 'AH [dB/km]': [0., 0.10, 11],  # [0., 0.20, 21]
                 # 'ZDR [dB]': [-0.8, 2.4, 17],  # [0., 0.20, 21]
                 'ZH+ [dBZ]': [-10, 60, 15], 'ZHa [dBZ]': [-10, 60, 15],
                 'ZH- [dBZ]': [-8, 8, 15],
                 },
    ptype=ptype, ucmap=ucmap, htiylim=[0, 12], htixlim=htixlim,
    cbticks=cbticks, contourl=contourl, tz=tz, fig_size=(19.2, 11),
    # unorm=unorm, ucmap=cmap_prabhakar,
    )
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)
plt.tight_layout()

if SAVE_FIGS:
    RES_DIR2 = LWDIR + f"pd_rdres/qpe_{DTWORK.strftime('%Y%m%d')}/ml_id/"
    if fullqc:
        RES_DIR2 += 'fullqc/'
    fname = (f"{DTWORK.strftime('%Y%m%d')}_{RADAR_SITE[:3].lower()}"
             + f"_daily_qvps_{v2p[:v2p.find('[')-1].lower()}.png")
    plt.savefig(RES_DIR2 + fname, format='png')
