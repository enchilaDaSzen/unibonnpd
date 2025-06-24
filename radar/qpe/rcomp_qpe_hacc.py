#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:20:19 2024

@author: dsanchez
"""

import datetime as dt
import os
import pickle
import numpy as np
from itertools import zip_longest

# =============================================================================
# Define working directory, time and list files
# =============================================================================
# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24h []
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24h [NO JXP]
START_TIME = dt.datetime(2018, 12, 2, 0, 0)  # 24 h [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24h [NO JXP]
# # START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 8, 0)  # 16h [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24h [NO BXP]

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# EWDIR = '/run/media/dsanchez/enchiladasz/safe/bonn_postdoc/'
LWDIR = '/home/enchiladaszen/Documents/sciebo/'
EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'
# EWDIR = '/media/enchiladaszen/enchiladasz/safe/bonn_postdoc/'

rcomp = 'rcomp_qpe_dwd'
rcomp = 'rcomp_qpe_dwdbxp'
# rcomp = 'rcomp_qpe_dwdjxp'
# rcomp = 'rcomp_qpe_dwdxpol'
# rcomp = 'rcomp_qpe_xpol'

qpe_amlb = False
if qpe_amlb:
    sffx = '_amlb'
else:
    sffx = ''

RQPE_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}/"
            + f"{rcomp}/5min{sffx}/")
RES_DIR = (EWDIR + f"pd_rdres/{START_TIME.strftime('%Y%m%d')}/"
           + f"{rcomp}/hourly{sffx}/")

rprods_dp = ['r_adp', 'r_ah', 'r_kdp', 'r_z']
rprods_hbr = ['r_ah_kdp', 'r_kdp_zdr', 'r_z_ah', 'r_z_kdp', 'r_z_zdr']
rprods_opt = ['r_kdpopt', 'r_zopt']
rprods_hyop = ['r_ah_kdpopt', 'r_zopt_ah', 'r_zopt_kdp', 'r_zopt_kdpopt']
rprods_nmd = ['r_aho_kdpo', 'r_kdpo', 'r_zo', 'r_zo_ah', 'r_zo_kdp',
              'r_zo_zdr']

rprods = sorted(rprods_dp + rprods_hbr + rprods_opt + rprods_hyop
                + ['r_zo', 'r_kdpo', 'r_aho_kdpo'])
# rprods = sorted(rprods_dp + rprods_hbr + rprods_opt + rprods_hyop
#                 + ['r_zo', 'r_kdpo', 'r_aho_kdpo', 'r_kdpo2'])

# %%
# =============================================================================
# Read in Radar QPE
# =============================================================================
def qpeg_reader(RQPE_GRID):
    """Read in the georeference data for the qpe."""
    for qpef in RQPE_GRID:
        if qpef.endswith('mgrid.tpy'):
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
# Create hourly acummulations
# =============================================================================
for rp in rprods:
    # print(f'{rp}.tpy')
    RQPE_FILES = [RQPE_DIR + i for i in sorted(os.listdir(RQPE_DIR))
                  if i.endswith(f'{rp}.tpy')
                  # if 'rcomp_rqpe' in i
                  ]
    RQPE_GRID = [RQPE_DIR + i for i in sorted(os.listdir(RQPE_DIR))
                 if i.endswith('mgrid.tpy')]
    RQPE_PARAMS = [RQPE_DIR + i for i in sorted(os.listdir(RQPE_DIR))
                   if i.endswith('params.tpy')]
    # qpe_georef = qpeg_reader(RQPE_GRID)
    qpe_params = qpeparams_reader(RQPE_PARAMS)
    # qpe_params = [i for i in qpep_reader(RQPE_FILES)]
    # with open(RES_DIR+qpe_params[0]['datetime'].strftime('%Y%m%d_')
    #           + 'params.tpy', 'wb') as f:
    #     pickle.dump(qpe_params, f, pickle.HIGHEST_PROTOCOL)
    ds_tres = dt.timedelta(minutes=5)
    ds_accum = dt.timedelta(hours=1)
    # nrays_comp = 2000
    # nrays_comp = 1000
    dsdt_full = [rdt['datetime'].replace(tzinfo=None) for rdt in qpe_params]
    # Here check that round works
    ds_accumg = round((dsdt_full[-1]-dsdt_full[0]+ds_tres)/ds_accum)
    ds_accumtg = int(ds_accum/ds_tres)
    ds_fullg = list(zip_longest(*(iter(enumerate(dsdt_full)),) * ds_accumtg))
    ds_fullg = [[itm for itm in l1 if itm is not None] for l1 in ds_fullg]
    # ds_fullgidx = [[(j[0]*nrays_comp, nrays_comp+j[0]*nrays_comp)
    #                 for j in i] for i in ds_fullg]

    # Accumulate using ds_accum
    rf_haccum = {}
    for cnt1, h_acc in enumerate(ds_fullg):
        # h_acc = ds_fullg[1]
        for cnt, idx in enumerate(h_acc):
            # print(idx)
            if cnt == 0:
                if RQPE_FILES[idx[0]].endswith(f'{rp}.tpy'):
                    with open(RQPE_FILES[idx[0]], 'rb') as fpkl:
                        resqpe = pickle.load(fpkl)
                        resqpe0 = {k: v for k, v in resqpe.items()
                                   if k.startswith('r_')}
            else:
                if RQPE_FILES[idx[0]].endswith(f'{rp}.tpy'):
                    with open(RQPE_FILES[idx[0]], 'rb') as fpkl:
                        resqpe = pickle.load(fpkl)
                        for k in resqpe0.keys():
                            resqpe0[k] = np.nansum((resqpe0[k], resqpe[k]),
                                                   axis=0)
        resqpe0 = {k: v/ds_accumtg for k, v in resqpe0.items()
                   if k.startswith('r_')}
        resqpe0['datetime'] = idx[1]
        resqpe0['elev_ang [deg]'] = resqpe['elev_ang [deg]']
        rf_haccum = {idx[1].strftime('%Y%m%d%H%M%S'): resqpe0}
        with open(RES_DIR+idx[1].strftime('%Y%m%d%H%M%S_')+f'rhqpe_{rp}.tpy',
                  'wb') as f:
            pickle.dump(rf_haccum, f, pickle.HIGHEST_PROTOCOL)
        print(idx[1].strftime('%Y%m%d%H') + f' [{rp}] ' + ' --- DONE')
