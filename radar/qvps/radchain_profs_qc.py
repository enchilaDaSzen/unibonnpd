#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:22:51 2022

@author: dsanchez
"""

import sys
import pickle
import numpy as np
from tqdm import tqdm
import towerpy as tp
import os
# import tad_phidp_offqvps as tpdp
# from copy import deepcopy
import datetime as dt
from zoneinfo import ZoneInfo
# from scipy import stats
# from towerpy.utils import radutilities as rut
import matplotlib.pyplot as plt
# import twpext as tpx
LWDIR = '/home/dsanchez/sciebo_dsr/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
sys.path.append(LWDIR + 'codes/github/unibonnpd/')
from radar.rparams_dwdxpol import RPARAMS

# =============================================================================
# Define working directory and list files
# =============================================================================
# Boxpol, Juxpol, Essen, Flechtdorf, Neuheilenbach, Offenthal, Hannover
RADAR_SITE = 'Offenthal'
PTYPE = 'qvps'

# DTWORK = dt.datetime(2016, 6, 4, 0, 0)
# DTWORK = dt.datetime(2017, 7, 19, 0, 0)
# DTWORK = dt.datetime(2017, 7, 24, 0, 0)
# DTWORK = dt.datetime(2017, 7, 25, 0, 0)
# DTWORK = dt.datetime(2018, 5, 16, 0, 0)
# DTWORK = dt.datetime(2018, 9, 23, 0, 0)
DTWORK = dt.datetime(2018, 12, 2, 0, 0)
# DTWORK = dt.datetime(2019, 5, 8, 0, 0)
# DTWORK = dt.datetime(2019, 5, 11, 0, 0)
# DTWORK = dt.datetime(2019, 7, 20, 0, 0)
# DTWORK = dt.datetime(2020, 6, 17, 0, 0)  # NO BXP
# DTWORK = dt.datetime(2021, 7, 13, 0, 0)
# DTWORK = dt.datetime(2021, 7, 14, 0, 0)
# DTWORK = dt.datetime(2019, 5, 10, 0, 0)
# DTWORK = dt.datetime(2019, 5, 11, 0, 0)
# DTWORK = dt.datetime(2021, 6, 20, 0, 0)  # 24 hr

EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

if 'xpol' in RADAR_SITE:
    WDIR = (EWDIR + f'pd_rdres/{PTYPE}/xpol/')
else:
    WDIR = (EWDIR + f'pd_rdres/{PTYPE}/dwd/')

PPFILES = [WDIR+i for i in sorted(os.listdir(WDIR))
           if i.endswith(f'_{PTYPE}.tpy') and RADAR_SITE in i
           and i.startswith(f"{DTWORK.strftime('%Y%m%d')}")]

extend_mlyr = False
if extend_mlyr:
    appx = '_extmlyr'
else:
    appx = ''

RPARAMS = {RADAR_SITE: next(item for item in RPARAMS
                            if item['site_name'] == RADAR_SITE)}

RES_DIR = LWDIR + f"pd_rdres/qvps_d4calib{appx}/{DTWORK.strftime('%Y%m%d')}/"
# RCFILES = [RES_DIR+i for i in sorted(os.listdir(RES_DIR))
#            if i.endswith(f'_{PTYPE}.tpy') and RADAR_SITE in i]

# %%
# =============================================================================
# Read radar profiles
# =============================================================================
with open(PPFILES[0], 'rb') as f:
    rprofs = pickle.load(f)

dtrs = [i.scandatetime for i in rprofs]
PLOT_METHODS = False
RESET = True

# %%
# =============================================================================
# ZH Offset adjustment
# =============================================================================
zh_oc = False
if zh_oc:
    if 'xpol' in RPARAMS[RADAR_SITE]['site_name']:
        zh_off = RPARAMS[RADAR_SITE]['zh_offset'].get(DTWORK.strftime("%Y%m%d"))
        # zh_off = 0
    else:
        zh_off = 0
    for rp in rprofs:
        rp.qvps['ZH [dBZ]'] += zh_off
    print(f'{RADAR_SITE}_ZH_O [{zh_off:.2f} dBZ]')

# =============================================================================
# PhiDP SHIFTED SIGN ADJUSTMENT
# =============================================================================
phidp_sign = False
if phidp_sign:
    if RADAR_SITE == 'Juxpol' and DTWORK.year > 2018:
        for cnt, rp in enumerate(rprofs):
            rp.qvps['PhiDP [deg]'] *= -1
    else:
        rp.qvps['PhiDP [deg]'] *= 1

# %%
# =============================================================================
# ML detection
# =============================================================================
qvps_res = np.median(np.diff(rprofs[0].georef['profiles_height [km]']))

mlid = {'minh': 1.5 if 'xpol' in RADAR_SITE else 1.,
        'maxh': 3.5 if 'xpol' in RADAR_SITE else 3.25,
        'kval': 0.08 if RADAR_SITE == 'Boxpol'
        else 0.08 if RADAR_SITE == 'Juxpol' else 0.08,
        'wval': 1/4 if RADAR_SITE == 'Boxpol' else 1/8
        if RADAR_SITE == 'Juxpol' else (3/4 if DTWORK.year < 2021 else 2/4),
        'zmin': 5 if 'xpol' in RADAR_SITE else 5,
        'zmax': 60 if 'xpol' in RADAR_SITE else 60,
        'comb': 14 if 'xpol' in RADAR_SITE else 14,
        'rhv': 0.85 if RADAR_SITE == 'Boxpol' else 0.85
        if RADAR_SITE == 'Juxpol' else 0.85,
        'ml_t': 0.85 if 'xpol' in RADAR_SITE else 0.85,
        'phidp': 'left'}
rmlyr = [tp.ml.mlyr.MeltingLayer(rd) for rd in rprofs]
[robj.ml_detection(rprofs[i], min_h=mlid['minh'], max_h=mlid['maxh'],
                   param_k=mlid['kval'], param_w=mlid['wval'],
                   comb_id=mlid['comb'], zhnorm_min=mlid['zmin'],
                   zhnorm_max=mlid['zmax'], rhvnorm_min=mlid['rhv'],
                   phidp_peak=mlid['phidp'])
 for i, robj in enumerate(tqdm(rmlyr, desc='rmlyr_towerpy'))]

print(f'QVPs resolution: {qvps_res:0.4f} [km]')
ml_thk = [i.ml_thickness for i in rmlyr]
mlid['ml_t'] = np.nanmean(ml_thk) + (0. if 'xpol' in RADAR_SITE else -0.)
print(f"ML_THK: {mlid['ml_t']:.2f}")

iprof = 0
iprof = 196
# iprof = 211
# iprof = 31
# iprof = 45

print(rprofs[iprof].scandatetime)
# rmlyr_i = tp.ml.mlyr.MeltingLayer(rprofs[iprof])
# rmlyr_i.ml_detection(rprofs[iprof], min_h=mlid['minh'], max_h=mlid['maxh'],
#                      param_k=mlid['kval'], param_w=mlid['wval'],
#                      comb_id=mlid['comb'], rhvnorm_min=mlid['rhv'],
#                      zhnorm_min=mlid['zmin'], zhnorm_max=mlid['zmax'],
#                      phidp_peak=mlid['phidp'], plot_method=True)

# =============================================================================
# Brief ML-PROCS and QC
# =============================================================================
for cnt, i in enumerate(rmlyr):
    # Remove unrealistic MLyrs
    if i.ml_bottom > i.ml_top:
        i.ml_bottom = np.nan
    if i.ml_top <= 0:
        i.ml_top = np.nan
    # Set a predefined MLyr thickness
    # if ~np.isnan(rmlyr[cnt].ml_top) and np.isnan(rmlyr[cnt].ml_bottom):
    #     i.ml_bottom = i.ml_top - mlid['ml_t']
    #     i.thickness = i.ml_top - i.ml_bottom
    # if np.isnan(rmlyr[cnt].ml_top) and ~np.isnan(rmlyr[cnt].ml_bottom):
    #     i.ml_top = i.ml_bottom + mlid['ml_t']
    #     i.thickness = i.ml_top - i.ml_bottom
    if i.ml_bottom <= 0.:
        # i.ml_bottom = 0.2
        i.ml_top = np.nan
    if i.ml_thickness <= 0.5:
        i.ml_bottom = i.ml_top - mlid['ml_t']
        # i.ml_top = i.ml_bottom + mlid['ml_t']
        i.thickness = i.ml_top - i.ml_bottom
    # Remove MLH using thresholds
    if rmlyr[cnt].ml_top < mlid['minh'] or rmlyr[cnt].ml_top > mlid['maxh']:
        i.ml_top = np.nan
        i.ml_bottom = np.nan
        i.thickness = np.nan
    # Double-check the ML thickness
    i.thickness = i.ml_top - i.ml_bottom

if not RESET:
    pi, pf = 0, 18
    a = np.random.rand(pf-pi)/10
    for c, i in enumerate(rmlyr[pi:pf]):
        # i.ml_top = 0.8358037542122864397873 + a[c]
        # i.ml_bottom = 0.5321241784185 - a[c]
        # i.thickness = i.ml_top - i.ml_bottom
        i.ml_top = 2.91251558704409628358037542122864397873 + a[c]
        i.ml_bottom = 2.0354182400886521545321241784185 - a[c]
        i.thickness = i.ml_top - i.ml_bottom
    for i in rmlyr[18:120]:
        i.ml_top = np.nan
        i.ml_bottom = np.nan
        i.thickness = i.ml_top - i.ml_bottom
    pi, pf = 120, 125
    a = np.random.rand(pf-pi)/10
    for c, i in enumerate(rmlyr[pi:pf]):
        # i.ml_top = 0.8358037542122864397873 + a[c]
        # i.ml_bottom = 0.5321241784185 - a[c]
        # i.thickness = i.ml_top - i.ml_bottom
        i.ml_top = 2.96143251558704409628358037542122864397873 + a[c]
        i.ml_bottom = 2.124182400886521545321241784185 - a[c]
        i.thickness = i.ml_top - i.ml_bottom

# %%

mov_avrgf_len = 5
mlb_mavf = np.ma.convolve(tp.utils.radutilities.fillnan1d([i.ml_bottom
                                                           for i in rmlyr]),
                          np.ones(mov_avrgf_len)/mov_avrgf_len, mode='same')
mlt_mavf = np.ma.convolve(tp.utils.radutilities.fillnan1d([i.ml_top
                                                           for i in rmlyr]),
                          np.ones(mov_avrgf_len)/mov_avrgf_len, mode='same')
rmlyr2 = [tp.ml.mlyr.MeltingLayer(rd) for rd in rprofs]
for cnt, i in enumerate(rmlyr2):
    if cnt < 2:  # mov_avrgf_len:
        i.ml_top = rmlyr[cnt].ml_top
        i.ml_bottom = rmlyr[cnt].ml_bottom
        i.ml_thickness = i.ml_top - i.ml_bottom
    elif cnt >= len(rmlyr2) - 2:  # mov_avrgf_len:
        i.ml_top = rmlyr[cnt].ml_top
        i.ml_bottom = rmlyr[cnt].ml_bottom
        i.ml_thickness = i.ml_top - i.ml_bottom
    else:
        i.ml_top = mlt_mavf[cnt]
        i.ml_bottom = mlb_mavf[cnt]
        i.ml_thickness = i.ml_top - i.ml_bottom
for cnt, i in enumerate(rmlyr2):
    # Remove unrealistic MLyrs
    if i.ml_bottom > i.ml_top:
        i.ml_top = np.nan
        i.ml_bottom = np.nan
        i.thickness = np.nan
    if i.ml_top <= 0:
        i.ml_top = np.nan
    if i.ml_bottom <= 0.:
        i.ml_bottom = 0.2
        # i.ml_bottom = 0.0
        # i.ml_bottom = np.nan
    # Remove flawed values that resulted from applying the MAF
    if np.isnan(rmlyr2[cnt].ml_top) and ~np.isnan(rmlyr2[cnt].ml_bottom):
        i.ml_top = np.nan
        i.ml_bottom = np.nan
        i.thickness = np.nan
    if not extend_mlyr:
        if np.isnan(rmlyr[cnt].ml_top) and np.isnan(rmlyr[cnt].ml_bottom):
            i.ml_top = np.nan
            i.ml_bottom = np.nan
            i.thickness = np.nan
    if ~np.isnan(rmlyr2[cnt].ml_top) and ~np.isnan(rmlyr2[cnt].ml_bottom):
        i.ml_top += (qvps_res*0. if RADAR_SITE == 'Boxpol' else
                     qvps_res*0. if RADAR_SITE == 'Juxpol' else
                     (qvps_res*0. if DTWORK.year < 2021 else qvps_res*0.))
        i.ml_bottom -= (qvps_res*0. if RADAR_SITE == 'Boxpol' else
                        qvps_res*0. if RADAR_SITE == 'Juxpol' else
                        (qvps_res*0. if DTWORK.year < 2021 else qvps_res*0.))
    # Remove unrealistic MLyrs
    # if i.ml_thickness <= mlid['ml_t'] or i.ml_thickness >= 3:
    if i.ml_thickness <= 0.5:
        i.ml_top = np.nan
        i.ml_bottom = np.nan
        i.thickness = np.nan
    # Double-check the ML thickness
    i.ml_thickness = i.ml_top - i.ml_bottom

if not RESET:
    pi, pf = 167, 181
    a = np.random.rand(pf-pi)/10
    for c, i in enumerate(rmlyr2[pi:pf]):
        i.ml_top = 2.251892122864397873 + a[c]
        i.ml_bottom = 1.1027 - a[c]
        i.thickness = i.ml_top - i.ml_bottom
    for i in rmlyr2[169:]:
        i.ml_top = np.nan
        i.ml_bottom = np.nan
        i.thickness = i.ml_top - i.ml_bottom
    for i in rmlyr2[144:151]:
        # i.ml_top = i.ml_top
        i.ml_bottom = i.ml_bottom
        i.thickness = i.ml_top - i.ml_bottom
    pi, pf = 187, 202
    a = np.random.rand(pf-pi)/10
    for c, i in enumerate(rmlyr2[pi:pf]):
        i.ml_top = 1.8851892122864397873 + a[c]
        i.ml_bottom = 0.9291027 - a[c]
        i.thickness = i.ml_top - i.ml_bottom


ml_top = [i.ml_top for i in rmlyr2]
print(f'ML_TOP (mean): {np.nanmean(ml_top):.2f}')
ml_btm = [i.ml_bottom for i in rmlyr2]
print(f'ML_BTM (mean): {np.nanmean(ml_btm):.2f}')
ml_thk = [i.ml_thickness for i in rmlyr2]
print(f'ML_THK (mean): {np.nanmean(ml_thk):.2f}')
# %%
# =============================================================================
# Profile Classification
# =============================================================================
min_h = (0.075 if 'xpol' in RADAR_SITE
         else (0.07 if DTWORK.year < 2021 else .1))
rhohv_thr_r = (0.80 if 'xpol' in RADAR_SITE
               else (0.80 if DTWORK.year < 2021 else 0.80))
rhohv_thr_p = (0.75 if 'xpol' in RADAR_SITE
               else (0.70 if DTWORK.year < 2021 else 0.75))
zh_thr_lr = 0
zh_thr_mr = 25
zh_thr_hr = 35
# zh_thr_mr = 30  # Too broad
minbins_lr = (4 if 'xpol' in RADAR_SITE else (1 if DTWORK.year < 2021 else 4))
minbins_mr = (5 if 'xpol' in RADAR_SITE else (2 if DTWORK.year < 2021 else 5))
minbins_hr = (2 if 'xpol' in RADAR_SITE else (1 if DTWORK.year < 2021 else 2))
for nprof, prf in enumerate(rprofs):
    # nprof = 217  # Use to evaluate a single profile
    # prf = rprofs[217]
    rr = rmlyr2[nprof].ml_bottom
    mlt = rmlyr2[nprof].ml_top
    # Convective-Type (no MLYR)
    if np.isnan(rr):
        # TODO: improve by checking rhoHV
        # Bins classification
        prf_class = np.where((prf.georef['profiles_height [km]'] > min_h)
                             & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
                             & (prf.qvps['ZH [dBZ]'] >= zh_thr_lr), 1, 0)
        prf_class = np.where(
            (prf.georef['profiles_height [km]'] > min_h)
            & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
            & (prf.qvps['ZH [dBZ]'] >= zh_thr_mr), 2, prf_class)
        prf_class = np.where(
            (prf.georef['profiles_height [km]'] > min_h)
            & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
            & (prf.qvps['ZH [dBZ]'] >= zh_thr_hr), 3, prf_class)
        prf.qvps['bin_class [0-5]'] = np.zeros_like(
            prf.qvps['ZH [dBZ]']) + prf_class
        prf.qvps['bin_class [0-5]'][np.isnan(prf.qvps['ZH [dBZ]'])] = np.nan
        prf_class = prf.qvps['bin_class [0-5]']
        # Profile classification
        # TODO: improve by measuring the height of the rain in the QVPs
        # if ((prf_class == 1).sum()*qvps_res) < 0.5:
        if ((prf_class == 0).sum()
                / np.count_nonzero(~np.isnan(prf_class))) > 0.5:
            prf.pcp_type = 0
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        # elif (prf_class == 1).sum() > 1:
        else:
            prf.pcp_type = 1 + 3  # 4
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        if (prf_class == 2).sum() >= minbins_mr:
            prf.pcp_type = 2 + 3  # 5
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        if (prf_class == 3).sum() >= minbins_hr:
            prf.pcp_type = 3 + 3  # 6
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        prf.qvps['prof_type [0-6]'][np.isnan(prf.qvps['ZH [dBZ]'])] = np.nan
    else:
        # Bins classification
        prf_class = np.where((prf.georef['profiles_height [km]'] >= min_h)
                             & (prf.georef['profiles_height [km]'] <= rr)
                             & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
                             & (prf.qvps['ZH [dBZ]'] >= zh_thr_lr),
                             1, 0)
        prf_class = np.where((prf.georef['profiles_height [km]'] >= min_h)
                             & (prf.georef['profiles_height [km]'] <= rr)
                             & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
                             & (prf.qvps['ZH [dBZ]'] >= zh_thr_mr),
                             2, prf_class)
        prf_class = np.where((prf.georef['profiles_height [km]'] >= min_h)
                             & (prf.georef['profiles_height [km]'] <= rr)
                             & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
                             & (prf.qvps['ZH [dBZ]'] >= zh_thr_hr),
                             3, prf_class)
        prf_class = np.where((prf.georef['profiles_height [km]'] > rr)
                             # & (prf.qvps['rhoHV [-]'] >= rhohv_thr_r)
                             & (prf.qvps['ZH [dBZ]'] >= zh_thr_lr)
                             & (prf.georef['profiles_height [km]'] <= mlt),
                             4, prf_class)
        prf_class = np.where((prf.georef['profiles_height [km]'] > mlt)
                             # & (prf.qvps['ZH [dBZ]'] > zh_thr_lr),
                             & (prf.qvps['rhoHV [-]'] >= rhohv_thr_p),
                             5, prf_class)
        prf.qvps['bin_class [0-5]'] = np.zeros_like(
            prf.qvps['ZH [dBZ]']) + prf_class
        prf.qvps['bin_class [0-5]'][np.isnan(prf.qvps['ZH [dBZ]'])] = np.nan
        prf_class = prf.qvps['bin_class [0-5]']
        # Profile classification
        # if ((prf_class == 1).sum()*qvps_res) < 0.5:
        if ((prf_class == 0).sum()
                / np.count_nonzero(~np.isnan(prf_class))) > 0.5:
            prf.pcp_type = 0
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        # elif (prf_class == 1).sum() > 1:
        else:
            prf.pcp_type = 1
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        if (prf_class == 2).sum() >= minbins_mr:
            prf.pcp_type = 2
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        if (prf_class == 3).sum() >= minbins_hr:
            prf.pcp_type = 3
            prf.qvps['prof_type [0-6]'] = np.ones_like(
                prf.qvps['ZH [dBZ]']) * prf.pcp_type
        prf.qvps['prof_type [0-6]'][np.isnan(prf.qvps['ZH [dBZ]'])] = np.nan

    prf.qvps.pop('prof_type [0-1]', None)
    prf.qvps.pop('class [0-1]', None)

# %%
min_h_zdr = (0.185 if 'xpol' in RADAR_SITE
             else (0.17 if DTWORK.year < 2021 else .20))
min_h_phidp = (0.175 if 'xpol' in RADAR_SITE
               else (0.2 if DTWORK.year < 2021 else 0.5))
zh_thr_lr = (20 if 'xpol' in RADAR_SITE
             else (25 if DTWORK.year < 2021 else 20))
# rhv_min = 0.985 if 'xpol' in RADAR_SITE else 0.95  # 0.95 old res 0.975
rhv_min = (0.985 if 'xpol' in RADAR_SITE
           else (0.97 if DTWORK.year < 2021 else 0.985))
min_bins = (6 if 'xpol' in RADAR_SITE
            else (2 if DTWORK.year < 2021 else 6))
maf_offset = True
# =============================================================================
# ZDR offset detection
# =============================================================================
roffzdr = [tp.calib.calib_zdr.ZDR_Calibration(rd) for rd in rprofs]
[robj.offsetdetection_qvps(pol_profs=rprofs[i], mlyr=rmlyr2[i],
                           min_h=min_h_zdr, zhmax=zh_thr_lr, rhvmin=rhv_min,
                           minbins=min_bins)
 for i, robj in enumerate(tqdm(roffzdr, desc='roffzdry_towerpy'))]

# zdro = np.array([robj.zdr_offset for i, robj in enumerate(roffzdr)])

# =============================================================================
# ZDR OFFSET QC
# =============================================================================
zdro = np.array([i.zdr_offset for i in roffzdr], dtype=float)
zdro[zdro == 0] = np.nan
if np.isnan(zdro[0]) and not np.isnan(zdro).all():
    zdro[0] = zdro[np.isfinite(zdro)][0]
if np.isnan(zdro).all():
    zdro[0] = 0.0

zdro = tp.utils.radutilities.fillnan1d(zdro)
if maf_offset:
    zdro_maf = np.ma.convolve(
        tp.utils.radutilities.fillnan1d([i for i in zdro]),
        np.ones(mov_avrgf_len)/mov_avrgf_len, mode='same')
    maw_adj = mov_avrgf_len - 2
    zdro[maw_adj:-maw_adj] = zdro_maf[maw_adj:-maw_adj]

if not RESET:
    zdro[zdro > -0.18] = -0.18

for cnt, robj in enumerate(roffzdr):
    robj.zdr_offset = zdro[cnt]

# %%
# =============================================================================
# PhiDP offset detection
# =============================================================================
# roffpdp = [tpdp.PhiDP_Calibration(rd) for rd in rprofs]
roffpdp = [tp.calib.calib_phidp.PhiDP_Calibration(rd) for rd in rprofs]
[robj.offsetdetection_qvps(
    pol_profs=rprofs[i], mlyr=rmlyr2[i], min_h=min_h_phidp, zhmax=zh_thr_lr,
    rhvmin=rhv_min, minbins=min_bins)
 for i, robj in enumerate(tqdm(roffpdp, desc='rcalphidpx_towerpy'))]

# =============================================================================
# PHIDP OFFSET QC
# =============================================================================
phidpo = np.array([i.phidp_offset for i in roffpdp], dtype=float)
# phidpo[:84] = np.nan
phidpo[phidpo == 0] = np.nan
if np.isnan(phidpo[0]) and not np.isnan(phidpo).all():
    phidpo[0] = phidpo[np.isfinite(phidpo)][0]
if np.isnan(phidpo).all():
    phidpo[0] = 0  # 144
phidpo = tp.utils.radutilities.fillnan1d(phidpo)
if maf_offset:
    phidpo_maf = np.ma.convolve(
        tp.utils.radutilities.fillnan1d([i for i in phidpo]),
        np.ones(mov_avrgf_len)/mov_avrgf_len, mode='same')
    phidpo[maw_adj:-maw_adj] = phidpo_maf[maw_adj:-maw_adj]

if not RESET:
    phidpo[phidpo < 170] = 0

for cnt, robj in enumerate(roffpdp):
    robj.phidp_offset = phidpo[cnt]
    # robj.phidp_sign = phidpsign[cnt]


# %%
# =============================================================================
# ZDR bias adjustment
# =============================================================================
zdr_oc = True
zdro = np.array([i.zdr_offset for i in roffzdr])

if zdr_oc:
    for cnt, rp in enumerate(rprofs):
        rp.qvps['ZDR [dB]'] -= zdro[cnt]

# =============================================================================
# PhiDP bias adjustment
# =============================================================================
phidp_oc = True
phidpo = np.array([i.phidp_offset for i in roffpdp])
if phidp_oc:
    for cnt, rp in enumerate(rprofs):
        rp.qvps['PhiDP [deg]'] -= phidpo[cnt]

# =============================================================================
# Adjust relative height
# =============================================================================
adjh = False
if adjh:
    RSITESH = {'Boxpol': 99.50, 'Juxpol': 310.00, 'Essen': 185.11,
               'Flechtdorf': 627.88, 'Neuheilenbach': 585.85,
               'Offenthal': 245.80}
    # Add rheight to mlyrs to work with hAMSL
    for ml in rmlyr2:
        ml.ml_top = ml.ml_top + RSITESH[RADAR_SITE]/1000
        ml.ml_bottom = ml.ml_bottom + RSITESH[RADAR_SITE]/1000
        ml.thickness = ml.ml_top - ml.ml_bottom
        # ml.ml_bottom += RSITESH[RADAR_SITE]/1000
    # Add rheight to profs to work with hAMSL
    for pr in rprofs:
        pr.georef['profiles_height [km]'] += RSITESH[RADAR_SITE]/1000

# %%
tz = 'Europe/Berlin'
htixlim = None
htixlim = [
    DTWORK.replace(tzinfo=ZoneInfo(tz)),
    (DTWORK + dt.timedelta(seconds=86399)).replace(tzinfo=ZoneInfo(tz))]
# dtm1 = [i.replace(tzinfo=None) for i in dtrs]

v2p = 'ZH [dBZ]'
# v2p = 'ZDR [dB]'
# v2p = 'rhoHV [-]'
# v2p = 'bin_class [0-5]'
# v2p = 'prof_type [0-6]'
# v2p='PhiDP [deg]'

pbins_class = {'no_rain': 0.5, 'light_rain': 1.5, 'modrt_rain': 2.5,
               'heavy_rain': 3.5, 'mixed_pcpn': 4.5, 'solid_pcpn': 5.5}
prof_type = {'NR': 0.5, 'LR [STR]': 1.5, 'MR [STR]': 2.5, 'HR [STR]': 3.5,
             'LR [CNV]': 4.5, 'MR [CNV]': 5.5, 'HR [CNV]': 6.5}


if v2p == 'bin_class [0-5]':
    ptype = 'pseudo'
    ucmap = 'tpylsc_rad_model'
    cbticks = pbins_class
elif v2p == 'prof_type [0-6]':
    ptype = 'pseudo'
    ucmap = 'coolwarm'
    ucmap = 'tpylsc_div_dbu_rd_r'
    ucmap = 'terrain'
    # ucmap = 'cividis'
    cbticks = prof_type
else:
    ptype = 'fcontour'
    # ptype = 'pseudo'
    ucmap = None
    cbticks = None

radb = tp.datavis.rad_interactive.hti_base(rprofs, mlyrs=rmlyr2,
                                           # stats='std_dev',
                                           var2plot=v2p,
                                           vars_bounds={'bin_class [0-5]':
                                                        (0, 6, 7),
                                                        'prof_type [0-6]':
                                                        (0, 7, 8)},
                                           # ptype=ptype, ucmap=ucmap,
                                           htiylim=[0, 12], htixlim=htixlim,
                                           cbticks=cbticks,
                                           # contourl='rhoHV [-]',
                                           # contourl='ZH [dBZ]',
                                           tz=tz, fig_size=(19.2, 11.4))
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)
plt.tight_layout()

# if PLOT_METHODS:
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=(True))
axs = ax[0]
axs.set_title('Offset variation using the QVPs method')
axs.plot([i.scandatetime for i in rprofs],
         np.array([i.zdr_offset for i in roffzdr]),
         marker='o', ms=5, mfc='None', label='QVPs data')
axs.grid(axis='y')
axs.tick_params(axis='both', labelsize=10)
axs.set_ylabel(r'$Z_{DR}$ [dB]', fontsize=10)
axs = ax[1]
axs.plot([i.scandatetime for i in rprofs],
         np.array([i.phidp_offset for i in roffpdp]),
         marker='o', ms=5, mfc='None', label='QVPs data')
axs.grid(axis='y')
axs.tick_params(axis='both', labelsize=10)
axs.set_ylabel(r'$\Phi_{DP}$ [deg]', fontsize=10)
axs.set_xlabel('Datetime', fontsize=10)
# plt.xlim([DTWORK, dt.datetime(2017, 7, 24, 23, 59)])
plt.xlim(htixlim)
# plt.ylim([-0.4, 0])
plt.tight_layout()
# %%
# =============================================================================
# Write the data objects into a file
# =============================================================================
# profs_cal = rprfc
elvp = rprofs[0].elev_angle
prof_pcp_type = np.array([i.pcp_type for i in rprofs])

profs_cal = {'phidpO': roffpdp, 'zdrO': roffzdr, 'mlyr': rmlyr2,
             'dtrs': [i.scandatetime for i in rprofs],
             'dtrs_ts': np.array([i.scandatetime.timestamp()
                                  for i in rprofs]),
             'pcp_type': prof_pcp_type}

fnamedt = (rprofs[0].scandatetime.strftime("%Y%m%d%H%M_") +
           rprofs[-1].scandatetime.strftime("%Y%m%d%H%M_"))
# fnamedt = (DTWORK.strftime("%Y%m%d%H%M_") +
#            rprofs[-1].scandatetime.strftime("%Y%m%d%H%M_"))

# # Save ML, offsets, etc
# with open(RES_DIR+fnamedt+RADAR_SITE+f'_qc_{elvp:.0f}{PTYPE}.tpy',
#           'wb') as f:
#     pickle.dump(profs_cal, f, pickle.HIGHEST_PROTOCOL)

# # Save QC-profiles
# with open(WDIR+f'qc{appx}/'+fnamedt+RADAR_SITE+f'{elvp:.0f}_qc_qvps.tpy',
#           'wb') as f:
#     pickle.dump(rprofs, f, pickle.HIGHEST_PROTOCOL)
