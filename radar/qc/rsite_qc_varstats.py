#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 12:37:32 2025

@author: dsanchez
"""

import datetime as dt
# from zoneinfo import ZoneInfo
import os
import pickle
import numpy as np
from scipy import stats
import towerpy as tp
# from towerpy.utils import unit_conversion as tpuc
# from towerpy.utils.radutilities import linspace_step
from radar import twpext as tpx
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
# from radar.rparams_dwdxpol import RPARAMS
from tqdm import tqdm

# =============================================================================
# Define working directory, and date-time
# =============================================================================

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sc1iebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'

# =============================================================================
# Define radar site
# =============================================================================
# Choose only one site at a time
# Boxpol, Juxpol, Essen, Flechtdorf, Neuheilenbach, Offenthal

# RSITES = ['Essen']

# =============================================================================
# Set plotting parameters
# =============================================================================
SAVE_FIGS = False
fig_size = (13, 7)
fig_size = (10.5, 7)

RES_DIR = LWDIR + 'pd_rdres/qpe_all/'

# =============================================================================
# Set QPE parameters
# =============================================================================
filter_aml = True
if filter_aml:
    appxf = ''
else:
    appxf = '_amlb'

# =============================================================================
# Define date-time
# =============================================================================
START_TIMES = [dt.datetime(2017, 7, 24, 0, 0),  # 24h [NO JXP]
               dt.datetime(2017, 7, 25, 0, 0),  # 24h [NO JXP]
               dt.datetime(2018, 5, 16, 0, 0),  # 24h []
               dt.datetime(2018, 9, 23, 0, 0),  # 24h [NO JXP]
               dt.datetime(2018, 12, 2, 0, 0),  # 24 h [NO JXP]
               dt.datetime(2019, 5, 8, 0, 0),   # 24h [NO JXP]
               dt.datetime(2019, 7, 20, 0, 0),  # 16h [NO BXP]
               dt.datetime(2020, 6, 17, 0, 0),  # 24h [NO BXP]
               dt.datetime(2021, 7, 13, 0, 0),  # 24h [NO BXP]
               dt.datetime(2021, 7, 14, 0, 0),  # 24h [NO BXP]
               ]

STOP_TIMES = [i+dt.timedelta(hours=24) for i in START_TIMES]

# %%

DMODE = 'read'
DSAVE = False
DDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/pd_rdres/qpe_all/'

if DMODE == 'compute':
    rvars_statsdaily = {}
    for START_TIME, STOP_TIME in zip(START_TIMES, STOP_TIMES):
        # print(START_TIME, STOP_TIME)
        if START_TIME > dt.datetime(2019, 5, 10, 0, 0):
            RSITES = ['Juxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach',
                      'Offenthal']
        elif START_TIME == dt.datetime(2018, 5, 16, 0, 0):
            RSITES = ['Boxpol', 'Juxpol', 'Essen', 'Flechtdorf',
                      'Neuheilenbach', 'Offenthal']
        else:
            RSITES = ['Boxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach',
                      'Offenthal']
        rvars_stats = {}
        for rcnt, RSITE in enumerate(
                tqdm(RSITES, desc=('Gathering radar vars -- '
                                   + START_TIME.strftime('%Y-%m-%d')))):
            # =============================================================================
            # List QC radar data
            # =============================================================================
            from radar.rparams_dwdxpol import RPARAMS
            RPARAMS = [next(item for item in RPARAMS
                            if item['site_name'] == RSITE)]
            suffix = ''  # _wrongbeta
            QCRD_DIR = {rs['site_name']:
                        EWDIR + (f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
                                 + f"/rsite_qc{suffix}/{rs['site_name']}/")
                        for rs in RPARAMS}
            RDQC_FILES = {k1: [i for i in sorted(os.listdir(rs))
                               if i.endswith('_rdqc.tpy')]
                          for k1, rs in QCRD_DIR.items()}
            # Check that date-time of the scans are within a given time window.
            rs_ts = {k1: np.array([dt.datetime.strptime(v2[:v2.find('_')],
                                                        '%Y%m%d%H%M%S%f')
                                   for v2 in v1])
                     for k1, v1 in RDQC_FILES.items()}
            rs_fts = {k1: tpx.fill_timeseries(rs_ts[k1],
                                              range(len(rs_ts[k1])),
                                              stspdt=(START_TIME, STOP_TIME),
                                              toldt=dt.timedelta(minutes=2))[1]
                      for k1, v1 in RDQC_FILES.items()}
            RDQC_FILES = {k1: [QCRD_DIR[k1]+RDQC_FILES[k1][i] if ~np.isnan(i)
                               else np.nan for i in rs]
                          for k1, rs in rs_fts.items()}
            RDQC_FILES = RDQC_FILES[RPARAMS[0]['site_name']]
            vars_stats = []
            for cnt, iraf in enumerate(RDQC_FILES):
                if iraf is not np.nan:
                    with open(iraf, 'rb') as f:
                        resattc = pickle.load(f)
                        rmlyr = tp.ml.mlyr.MeltingLayer(resattc)
                        rmlyr.ml_top = resattc.ml_top
                        rmlyr.ml_bottom = resattc.ml_bottom
                        rmlyr.ml_thickness = resattc.ml_thickness
                        # PPI MLYR
                        rmlyr.ml_ppidelimitation(resattc.georef,
                                                 resattc.params, resattc.vars)
                        rband = next(item['rband'] for item in RPARAMS
                                     if item['site_name'] == resattc.site_name)
                        nvars2use = ['ZH [dBZ]', 'ZH+ [dBZ]', 'ZDR [dB]',
                                     'AH [dB/km]', 'ADP [dB/km]',
                                     'KDP [deg/km]', 'KDP+ [deg/km]']
                        # nvars2use = ['ZH+ [dBZ]']
                        if filter_aml:
                            varsdata = {nvar: np.where(
                                (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                                resattc.vars[nvar], np.nan)
                                for nvar in nvars2use}
                        else:
                            varsdata = {nvar: resattc.vars[nvar]
                                        for nvar in nvars2use}
                        vars_statsi = {
                            nvar:
                                {'mean': np.nanmean(varsdata[nvar]),
                                 'med': np.nanmedian(varsdata[nvar]),
                                 # 'std': np.nanstd(varsdata[nvar], axis=-1,
                                 #                  ddof=1),
                                 # 'var': np.nanvar(varsdata[nvar], axis=-1,
                                 #                  ddof=1),
                                 # 'mode': stats.mode(varsdata[nvar],
                                 #                    nan_policy='omit'),
                                 'min': np.nanmin(varsdata[nvar]),
                                 'max': np.nanmax(varsdata[nvar])}
                                for nvar in nvars2use}
                        vars_stats.append(vars_statsi)
            rvars_stats[RSITE] = vars_stats
        rvars_statsdaily[START_TIME.strftime('%Y%m%d')] = rvars_stats

    if DSAVE:
        with open(f'{DDIR}rvars_statsdaily.tpy', 'wb') as f:
            pickle.dump(rvars_statsdaily, f, pickle.HIGHEST_PROTOCOL)

if DMODE == 'read':
    with open(f'{DDIR}rvars_statsdaily.tpy', 'rb') as f:
        rvars_statsdaily = pickle.load(f)
# %%

RSITES = ['Boxpol', 'Juxpol', 'Essen', 'Flechtdorf', 'Neuheilenbach',
          'Offenthal']

stat2plot = 'mean'
nvars2use = 'ZH+ [dBZ]'

norm = mpc.BoundaryNorm(np.linspace(-10, 60, 15),
                        mpl.colormaps['tpylsc_rad_ref'].N, extend='both')

rstats = {'mean': [{rn: np.nanmean([iscan[nvars2use][stat2plot]
                                    for iscan in rdayv[rn]])
                    if rn in rdayv.keys() else np.nan for rn in RSITES}
                   for rdayn, rdayv in rvars_statsdaily.items()],
          'med': [{rn: np.nanmedian([iscan[nvars2use][stat2plot]
                                     for iscan in rdayv[rn]])
                   if rn in rdayv.keys() else np.nan for rn in RSITES}
                  for rdayv in rvars_statsdaily.values()],
          'max': [{rn: np.nanmax([iscan[nvars2use][stat2plot]
                                  for iscan in rdayv[rn]])
                   if rn in rdayv.keys() else np.nan for rn in RSITES}
                  for rdayv in rvars_statsdaily.values()],
          'min': [{rn: np.nanmin([iscan[nvars2use][stat2plot]
                                  for iscan in rdayv[rn]])
                   if rn in rdayv.keys() else np.nan for rn in RSITES}
                  for rdayv in rvars_statsdaily.values()]}

fig, ax = plt.subplots(figsize=(18.6, 5))

im, cbar = tpx.heatmap(np.array([np.array([v1 for v1 in i1.values()])
                                 for i1 in rstats[stat2plot]]).T, RSITES,
                       [scdt.strftime('%Y-%m-%d') for scdt in START_TIMES],
                       aspect='auto', ax=ax, cmap="tpylsc_rad_ref", norm=norm,
                       cbarlabel=r"$\overline{Z_H}$ [dBZ]",
                       cbar_kw={'extend': 'both', })
texts = tpx.annotate_heatmap(im, valfmt="{x:d}", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout()
plt.show()

if SAVE_FIGS:
    len_rdtsets = len(START_TIMES)
    RES_DIR2 = RES_DIR + 'rcomp_qpe_dwd_dwdxpol/'
    fname = ('dailyscans_'
             + f"{nvars2use[:nvars2use.find('[')-1].replace('+', 'P')}"
             + f"{stat2plot}.png")
    plt.savefig(RES_DIR2 + fname, dpi=200, format='png')
