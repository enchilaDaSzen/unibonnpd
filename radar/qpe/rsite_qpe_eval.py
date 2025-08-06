#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:09:09 2024

@author: dsanchez
"""

import datetime as dt
from zoneinfo import ZoneInfo
import os
import pickle
import numpy as np
import towerpy as tp
from itertools import zip_longest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from towerpy.datavis import rad_display
from radar import twpext as tpx
from radar.rparams_dwdxpol import RPARAMS, RPRODSLTX
from tqdm import tqdm

# =============================================================================
# Define working directory, and date-time
# =============================================================================

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

# START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24 h [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24 h [NO JXP]
# START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24 h []
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24 h [NO JXP]
START_TIME = dt.datetime(2018, 12, 2, 0, 0)  # 24 h [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24 h [NO JXP]
# # START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 24 h [NO JXP]
START_TIME = dt.datetime(2019, 7, 20, 8, 0)  # 16 h [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24 h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24 h [NO BXP]

# STOP_TIME = dt.datetime(2021, 7, 13, 23, 59)
# EVNTD_HRS = round((STOP_TIME - START_TIME).total_seconds() / 3600)
EVNTD_HRS = (16 if START_TIME == dt.datetime(2019, 7, 20, 8, 0) else 24)

STOP_TIME = START_TIME + dt.timedelta(hours=EVNTD_HRS)
QPE_TRES = dt.timedelta(minutes=5)

# =============================================================================
# Define radar site
# =============================================================================
# Choose only one site at a time
# Boxpol, Juxpol, Essen, Flechtdorf, Neuheilenbach, Offenthal
RSITE = 'Juxpol'
RPARAMS = [next(item for item in RPARAMS if item['site_name'] == RSITE)]

# =============================================================================
# Read-in QVPs data
# =============================================================================
data4calib = 'qvps'
DIRPROFSCAL = LWDIR + f"pd_rdres/qvps_d4calib/{START_TIME.strftime('%Y%m%d')}/"

RCAL_FILES = {RSITE: DIRPROFSCAL+n for n in sorted(os.listdir(DIRPROFSCAL))
              if data4calib in n and RSITE in n}

with open(RCAL_FILES[RSITE], 'rb') as breader:
    profs_data = pickle.load(breader)

mlyrhv = [i for i in profs_data['mlyr'] if ~np.isnan(i.ml_top)]
mlt_avg = np.nanmean([i.ml_top for i in profs_data['mlyr']])
mlk_avg = np.nanmean([i.ml_thickness for i in profs_data['mlyr']])
mlb_avg = np.nanmean([i.ml_bottom for i in profs_data['mlyr']])
phidpOv = [i for i in profs_data['phidpO'] if ~np.isnan(i.phidp_offset)]
zdrOv = [i for i in profs_data['zdrO'] if ~np.isnan(i.zdr_offset)]

# =============================================================================
# Set plotting parameters
# =============================================================================
PLOT_METHODS = False
PLOT_FIGS = True
SAVE_FIGS = True
SAVE_COEFFS = False

# fig_size = (13.5, 7)
fig_size = (10.5, 7)

# xlims, ylims = [4.3, 9.2], [48.75, 52.75]  # XPOL NRW
xlims, ylims = [4.324, 10.953], [48.635, 52.754]  # DWDXPOL RADCOV
xlims, ylims = [5.85, 11.], [48.55, 52.75]  # DWDXPOL DE

RES_DIR = LWDIR + f"pd_rdres/qpe_{START_TIME.strftime('%Y%m%d')}/rsite_qpe/"

if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
    RPRODSLTX['r_kdpo'] = '$R(K_{DP})[OV]$'
    RPRODSLTX['r_zo'] = '$R(Z_{H})[OA]$'
    RPRODSLTX['r_aho_kdpo'] = '$R(A_{H}, K_{DP})[OV]$'
# %%
# =============================================================================
# List QC radar data
# =============================================================================
suffix = ''  # _wrongbeta
QCRD_DIR = {rs['site_name']:
            EWDIR + (f"pd_rdres/{START_TIME.strftime('%Y%m%d')}"
                     + f"/rsite_qc{suffix}/{rs['site_name']}/")
            for rs in RPARAMS}

RDQC_FILES = {k1: [i for i in sorted(os.listdir(rs))
                   if i.endswith('_rdqc.tpy')] for k1, rs in QCRD_DIR.items()}

# Check that date-time of the scans are within a given time window.
rs_ts = {k1: np.array([dt.datetime.strptime(v2[:v2.find('_')],
                                            '%Y%m%d%H%M%S%f')
                       for v2 in v1]) for k1, v1 in RDQC_FILES.items()}
rs_fts = {k1: tpx.fill_timeseries(rs_ts[k1],
                                  range(len(rs_ts[k1])),
                                  stspdt=(START_TIME, STOP_TIME),
                                  toldt=dt.timedelta(minutes=2))[1]
          for k1, v1 in RDQC_FILES.items()}

RDQC_FILES = {k1: [QCRD_DIR[k1]+RDQC_FILES[k1][i] if ~np.isnan(i)
                   else np.nan for i in rs] for k1, rs in rs_fts.items()}

RDQC_FILES = RDQC_FILES[RPARAMS[0]['site_name']]

ds_accum = dt.timedelta(hours=1)
dsdt_full = np.arange(START_TIME, STOP_TIME, QPE_TRES).astype(dt.datetime)
ds_accumg = round((dsdt_full[-1]-dsdt_full[0]+QPE_TRES)/ds_accum)
ds_accumtg = int(ds_accum/QPE_TRES)
ds_fullg = list(zip_longest(*(iter(enumerate(dsdt_full)),) * ds_accumtg))
ds_fullg = [[itm for itm in l1 if itm is not None] for l1 in ds_fullg]
ds_fullgidx = [[(j[0], RDQC_FILES[j[0]]) for j in i] for i in ds_fullg]

# %%
# =============================================================================
# Set QPE parameters
# =============================================================================
if mlb_avg > 1.5:
    temp = 15
else:
    # temp = 7.5
    temp = 15
z_thld = 40
qpe_amlb = False

if qpe_amlb:
    appxf = '_amlb'
else:
    appxf = ''

rprods_dp = ['r_adp', 'r_ah', 'r_kdp', 'r_z']
rprods_hbr = ['r_ah_kdp', 'r_kdp_zdr', 'r_z_ah', 'r_z_kdp', 'r_z_zdr']
rprods_opt = ['r_kdpopt', 'r_zopt']
rprods_hyop = ['r_ah_kdpopt', 'r_zopt_ah', 'r_zopt_kdp', 'r_zopt_kdpopt']
rprods_nmd = ['r_aho_kdpo', 'r_kdpo', 'r_zo', 'r_zo_ah', 'r_zo_kdp',
              'r_zo_zdr']

rprods = sorted(rprods_dp[1:] + rprods_hbr[1:] + rprods_opt + rprods_hyop
                + ['r_zo', 'r_kdpo', 'r_aho_kdpo'])

# rprods = sorted(['r_ah'] + rprods_opt)

# rprods = sorted(['r_ah', 'r_z', 'r_zopt'])
# %%
# =============================================================================
# Read-in QC PPI rdata
# =============================================================================
rqpe_dt = []
mlyrh = []
r_minsnr = []
rphidp0 = []
rzdr0 = []
radapt_coeffs = {'z_opt': [], 'kdp_opt': []}

rqpe_acch = [{rp: 0 for rp in rprods} for h1 in ds_fullgidx]
vld1 = [cnt for cnt, i1 in enumerate(RDQC_FILES) if type(i1) is str][0]

for cnt, fhrad in enumerate(tqdm(ds_fullgidx,
                                 desc=f'Computing RQPE [{RSITE}]')):
    for frad in fhrad:
        if frad[1] is not np.nan:
            # frad = [0, ds_fullgidx[17][6][1]]  # USE FOR ANALYSIS
            with open(frad[1], 'rb') as f:
                resattc = pickle.load(f)
                rmlyr = tp.ml.mlyr.MeltingLayer(resattc)
                rmlyr.ml_top = resattc.ml_top
                rmlyr.ml_bottom = resattc.ml_bottom
                rmlyr.ml_thickness = resattc.ml_thickness
                # PPI MLYR
                rmlyr.ml_ppidelimitation(resattc.georef, resattc.params,
                                         resattc.vars)
                mlyrh.append([rmlyr.ml_top, rmlyr.ml_thickness,
                              rmlyr.ml_bottom])
                r_minsnr.append(resattc.min_snr)
                rphidp0.append(resattc.phidp0)
                rzdr0.append(resattc.zdr_offset)
                rband = next(item['rband'] for item in RPARAMS
                             if item['site_name'] == resattc.site_name)
                # TODO: FOR C BAND ZH(ATTC) WORKS BETTER FOR KDP, WHY?
                if rband == 'C':
                    # zh4rkdpo = 'ZH [dBZ]'  # ZH(ATTC)
                    zh4rkdpo = 'ZH+ [dBZ]'  # ZH(AH)
                else:
                    zh4rkdpo = 'ZH+ [dBZ]'  # ZH(AH)
                zh4rzo = 'ZH+ [dBZ]'  # ZH(AH)
                zh4r = 'ZH+ [dBZ]'  # ZH(AH)
                zdr4r = 'ZDR [dB]'
                ah4r = 'AH [dB/km]'
                adpr = 'ADP [dB/km]'
                # kdp4rkdpo = 'KDP [deg/km]'  # AH
                kdp4rkdpo = 'KDP+ [deg/km]'  # Vulpiani+AH
                kdp4r = 'KDP+ [deg/km]'  # Vulpiani+AH
                if rband == 'C':
                    if START_TIME == dt.datetime(2021, 7, 14, 0, 0):
                        rz_a, rz_b = (1/0.026)**(1/0.69), 1/0.69  # Chen2023
                        rkdp_a, rkdp_b = 30.6, 0.71  # Chen2023
                        rah_a, rah_b = 427, 0.94  # Chen2023
                    else:
                        rz_a, rz_b = (1/0.052)**(1/0.57), 1/0.57  # Chen2021
                        rah_a, rah_b = 307, 0.92  # Chen2021
                        rkdp_a, rkdp_b = 20.7, 0.72  # Chen2021
                elif rband == 'X':
                    if START_TIME == dt.datetime(2021, 7, 14, 0, 0):
                        rz_a, rz_b = (1/0.057)**(1/0.57), 1/0.57  # Chen2023
                        rkdp_a, rkdp_b = 22.9, 0.76  # Chen2023
                        rah_a, rah_b = 67, 0.78  # Chen2023
                    else:
                        rz_a, rz_b = (1/0.098)**(1/0.47), 1/0.47  # Chen2021
                        rah_a, rah_b = 38, 0.69  # Chen2021
                        rkdp_a, rkdp_b = 15.6, 0.83  # Chen2021
            # =============================================================================
            # Rainfall estimators
            # =============================================================================
                rqpe = tp.qpe.qpe_algs.RadarQPE(resattc)
                if 'r_adp' in rprods:
                    rqpe.adp_to_r(
                        resattc.vars[adpr], mlyr=rmlyr, temp=temp, rband=rband,
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_ah' in rprods:
                    rqpe.ah_to_r(
                        resattc.vars[ah4r], mlyr=rmlyr, temp=temp, rband=rband,
                        # a=rah_a, b=rah_b,
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_kdp' in rprods:
                    rqpe.kdp_to_r(
                        resattc.vars[kdp4r], mlyr=rmlyr,
                        a=next(item['rkdp_a'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        b=next(item['rkdp_b'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_z' in rprods:
                    rqpe.z_to_r(resattc.vars[zh4r], mlyr=rmlyr,
                                a=next(item['rz_a'] for item in RPARAMS
                                       if item['site_name'] == rqpe.site_name),
                                b=next(item['rz_b'] for item in RPARAMS
                                       if item['site_name'] == rqpe.site_name),
                                beam_height=resattc.georef['beam_height [km]'])
            # =============================================================================
            # Hybrid estimators
            # =============================================================================
                if 'r_kdp_zdr' in rprods:
                    rqpe.kdp_zdr_to_r(
                        resattc.vars[kdp4r], resattc.vars[zdr4r], mlyr=rmlyr,
                        a=next(item['rkdpzdr_a'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        b=next(item['rkdpzdr_b'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        c=next(item['rkdpzdr_c'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_z_ah' in rprods:
                    rqpe.z_ah_to_r(
                        resattc.vars[zh4r], resattc.vars[ah4r], mlyr=rmlyr,
                        z_thld=z_thld, temp=temp, rband=rband,
                        rz_a=next(item['rz_a'] for item in RPARAMS
                                  if item['site_name'] == rqpe.site_name),
                        rz_b=next(item['rz_b'] for item in RPARAMS
                                  if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_z_kdp' in rprods:
                    rqpe.z_kdp_to_r(
                        resattc.vars[zh4r], resattc.vars[kdp4r], mlyr=rmlyr,
                        z_thld=z_thld,
                        rz_a=next(item['rz_a'] for item in RPARAMS
                                  if item['site_name'] == rqpe.site_name),
                        rz_b=next(item['rz_b'] for item in RPARAMS
                                  if item['site_name'] == rqpe.site_name),
                        rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                                    if item['site_name'] == rqpe.site_name),
                        rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                                    if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_z_zdr' in rprods:
                    rqpe.z_zdr_to_r(
                        resattc.vars[zh4r], resattc.vars[zdr4r], mlyr=rmlyr,
                        a=next(item['rzhzdr_a'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        b=next(item['rzhzdr_b'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        c=next(item['rzhzdr_c'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                if 'r_ah_kdp' in rprods:
                    rqpe.ah_kdp_to_r(
                        resattc.vars[zh4r], resattc.vars[ah4r],
                        resattc.vars[kdp4r], mlyr=rmlyr, z_thld=z_thld,
                        temp=temp, rband=rband,
                        rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                                    if item['site_name'] == rqpe.site_name),
                        rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                                    if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
            # =============================================================================
            # Adaptive estimators
            # =============================================================================
                rqpe_opt = tp.qpe.qpe_algs.RadarQPE(resattc)
                if 'r_kdpopt' in rprods:
                    rkdp_fit = tpx.rkdp_opt(
                        resattc.vars[kdp4rkdpo], resattc.vars[zh4rkdpo],
                        mlyr=rmlyr, rband=rband, kdpmed=0.5,
                        zh_thr=((40, 50) if rband == 'X' else (44.5, 45.5)),
                        rkdp_stv=(next(item['rkdp_a'] for item in RPARAMS
                                       if item['site_name'] == rqpe.site_name),
                                  next(item['rkdp_b'] for item in RPARAMS
                                       if item['site_name'] == rqpe.site_name)))
                    rqpe_opt.kdp_to_r(
                        resattc.vars[kdp4r], a=rkdp_fit[0], b=rkdp_fit[1],
                        mlyr=rmlyr,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_kdpopt = rqpe_opt.r_kdp
                    radapt_coeffs['kdp_opt'].append(
                        (((rkdp_fit[0], rkdp_fit[1]))))
                if 'r_zopt' in rprods:
                    rzh_fit = tpx.rzh_opt(
                        resattc.vars[zh4rzo], rqpe.r_ah, resattc.vars[ah4r],
                        pia=resattc.vars['PIA [dB]'], mlyr=rmlyr,
                        # maxpia=(50 if rband == 'X' else 20),
                        # fit_ab=True,
                        minpia=(.1 if resattc.site_name == 'Juxpol' else .1),
                        rzfit_b=(2.14 if rband == 'X' else 1.6),
                        rz_stv=[[next(item['rz_a'] for item in RPARAMS
                                     if item['site_name'] == rqpe.site_name),
                                next(item['rz_b'] for item in RPARAMS
                                     if item['site_name'] == rqpe.site_name)]],
                        plot_method=PLOT_METHODS)
                    rqpe_opt.z_to_r(
                        resattc.vars[zh4r], mlyr=rmlyr, a=rzh_fit[0],
                        b=rzh_fit[1],
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zopt = rqpe_opt.r_z
                    radapt_coeffs['z_opt'].append(((
                                          (rqpe.r_zopt['coeff_a'],
                                           rqpe.r_zopt['coeff_b']))))
                if 'r_zopt_ah' in rprods and 'r_zopt' in rprods:
                    rqpe_opt.z_ah_to_r(
                        resattc.vars[zh4r], resattc.vars[ah4r], mlyr=rmlyr,
                        z_thld=z_thld, temp=temp, rband=rband,
                        rz_a=rzh_fit[0], rz_b=rzh_fit[1],
                        # rah_a=67, rah_b=0.78,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zopt_ah = rqpe_opt.r_z_ah
                if 'r_zopt_kdp' in rprods and 'r_zopt' in rprods:
                    rqpe_opt.z_kdp_to_r(
                        resattc.vars[zh4r], resattc.vars[kdp4r], mlyr=rmlyr,
                        z_thld=z_thld, rz_a=rzh_fit[0], rz_b=rzh_fit[1],
                        rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                                    if item['site_name'] == rqpe.site_name),
                        rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                                    if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zopt_kdp = rqpe_opt.r_z_kdp
                rqpe_opt2 = tp.qpe.qpe_algs.RadarQPE(resattc)
                if ('r_zopt_kdpopt' in rprods and 'r_zopt' in rprods
                        and 'r_kdpopt' in rprods):
                    rqpe_opt2.z_kdp_to_r(
                        resattc.vars[zh4r], resattc.vars[kdp4r], mlyr=rmlyr,
                        z_thld=z_thld, rz_a=rzh_fit[0], rz_b=rzh_fit[1],
                        rkdp_a=rkdp_fit[0], rkdp_b=rkdp_fit[1],
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zopt_kdpopt = rqpe_opt2.r_z_kdp
                if ('r_ah_kdpopt' in rprods and 'r_kdpopt' in rprods):
                    rqpe_opt2.ah_kdp_to_r(
                        resattc.vars[zh4r], resattc.vars[ah4r],
                        resattc.vars[kdp4r], mlyr=rmlyr, z_thld=z_thld,
                        temp=temp, rband=rband,
                        # rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                        #             if item['site_name'] == rqpe.site_name),
                        # rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                        #             if item['site_name'] == rqpe.site_name),
                        rkdp_a=rkdp_fit[0], rkdp_b=rkdp_fit[1],
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_ah_kdpopt = rqpe_opt2.r_ah_kdp
            # =============================================================================
            # Estimators using non-fully corrected variables
            # =============================================================================
                rqpe_nfc = tp.qpe.qpe_algs.RadarQPE(resattc)
                zh4r2 = 'ZH [dBZ]'  # ZHattc
                kdp4r2 = 'KDP* [deg/km]'  # Vulpiani
                if 'r_zo' in rprods:
                    rqpe_nfc.z_to_r(
                        resattc.vars[zh4r2], mlyr=rmlyr, a=rz_a, b=rz_b,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zo = rqpe_nfc.r_z
                if 'r_kdpo' in rprods:
                    rqpe_nfc.kdp_to_r(
                        resattc.vars[kdp4r2], mlyr=rmlyr, a=rkdp_a, b=rkdp_b,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_kdpo = rqpe_nfc.r_kdp
                if 'r_zo_ah' in rprods:
                    rqpe_nfc.z_ah_to_r(
                        resattc.vars[zh4r2], resattc.vars[ah4r], rz_a=rz_a,
                        rz_b=rz_b, rah_a=rah_a, rah_b=rah_b, z_thld=z_thld,
                        mlyr=rmlyr, rband=rband,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zo_ah = rqpe_nfc.r_z_ah
                if 'r_zo_kdp' in rprods:
                    rqpe_nfc.z_kdp_to_r(
                        resattc.vars[zh4r2], resattc.vars[kdp4r2], rz_a=rz_a,
                        rz_b=rz_b, rkdp_a=rkdp_a, rkdp_b=rkdp_b, z_thld=z_thld,
                        mlyr=rmlyr,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zo_kdp = rqpe_nfc.r_z_kdp
                if 'r_zo_zdr' in rprods:
                    rqpe_nfc.z_zdr_to_r(
                        resattc.vars[zh4r2], resattc.vars[zdr4r], mlyr=rmlyr,
                        a=next(item['rzhzdr_a'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        b=next(item['rzhzdr_b'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        c=next(item['rzhzdr_c'] for item in RPARAMS
                               if item['site_name'] == rqpe.site_name),
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_zo_zdr = rqpe_nfc.r_z_zdr
                if 'r_aho_kdpo' in rprods:
                    rqpe_nfc.ah_kdp_to_r(
                        resattc.vars[zh4r2], resattc.vars[ah4r],
                        resattc.vars[kdp4r2], mlyr=rmlyr, temp=temp,
                        z_thld=z_thld, rband=rband,
                        rah_a=rah_a, rah_b=rah_b, rkdp_a=rkdp_a, rkdp_b=rkdp_b,
                        beam_height=resattc.georef['beam_height [km]'])
                    rqpe.r_aho_kdpo = rqpe_nfc.r_ah_kdp
            # =============================================================================
            # QPE within and above the MLYR
            # =============================================================================
                if qpe_amlb:
                    thr_zwsnw = next(item['thr_zwsnw'] for item in RPARAMS
                                     if item['site_name'] == rqpe.site_name)
                    thr_zhail = next(item['thr_zhail'] for item in RPARAMS
                                     if item['site_name'] == rqpe.site_name)
                    f_rz_ml = next(item['f_rz_ml'] for item in RPARAMS
                                   if item['site_name'] == rqpe.site_name)
                    f_rz_sp = next(item['f_rz_sp'] for item in RPARAMS
                                   if item['site_name'] == rqpe.site_name)
                else:
                    thr_zwsnw = next(item['thr_zwsnw'] for item in RPARAMS
                                     if item['site_name'] == rqpe.site_name)
                    thr_zhail = next(item['thr_zhail'] for item in RPARAMS
                                     if item['site_name'] == rqpe.site_name)
                    f_rz_ml = 0
                    f_rz_sp = 0
                # additional factor for the RZ relation is applied to data
                # within the ML
                rqpe_ml = tp.qpe.qpe_algs.RadarQPE(resattc)
                rqpe_ml.z_to_r(resattc.vars[zh4r],
                               a=next(item['rz_a'] for item in RPARAMS
                                      if item['site_name'] == rqpe.site_name),
                               b=next(item['rz_b'] for item in RPARAMS
                                      if item['site_name'] == rqpe.site_name))
                rqpe_ml.r_z['Rainfall [mm/h]'] = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 2)
                    & (resattc.vars[zh4r] > thr_zwsnw),
                    rqpe_ml.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)
                # additional factor for the RZ relation is applied to data
                # above the ML
                rqpe_sp = tp.qpe.qpe_algs.RadarQPE(resattc)
                rqpe_sp.z_to_r(resattc.vars[zh4r],
                               a=next(item['rz_a'] for item in RPARAMS
                                      if item['site_name'] == rqpe.site_name),
                               b=next(item['rz_b'] for item in RPARAMS
                                      if item['site_name'] == rqpe.site_name))
                rqpe_sp.r_z['Rainfall [mm/h]'] = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 3.),
                    rqpe_sp.r_z['Rainfall [mm/h]']*f_rz_sp,
                    rqpe_ml.r_z['Rainfall [mm/h]'])
                # Correct all other variables
                [setattr(
                    rqpe, rp, {(k1): (np.where(
                        (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                        getattr(rqpe, rp)['Rainfall [mm/h]'],
                        rqpe_sp.r_z['Rainfall [mm/h]']) if 'Rainfall' in k1
                        else v1) for k1, v1 in getattr(rqpe, rp).items()})
                    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]
                # rz_hail is applied to data below the ML with Z > 55 dBZ
                rqpe_hail = tp.qpe.qpe_algs.RadarQPE(resattc)
                rqpe_hail.z_to_r(
                    resattc.vars[zh4r],
                    a=next(item['rz_haila'] for item in RPARAMS
                           if item['site_name'] == rqpe.site_name),
                    b=next(item['rz_hailb'] for item in RPARAMS
                           if item['site_name'] == rqpe.site_name))
                rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
                    & (resattc.vars[zh4r] >= thr_zhail),
                    rqpe_hail.r_z['Rainfall [mm/h]'], 0)
                # Set a limit in range
                # max_rkm = 270.
                # grid_range = (
                #     np.ones_like(resattc.georef['beam_height [km]'])
                #     * resattc.georef['range [m]']/1000)
                # rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
                #     (grid_range > max_rkm), 0,
                #     rqpe_hail.r_z['Rainfall [mm/h]'])
                # rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
                #     (np.isnan(rqpe.r_z['Rainfall [mm/h]'])), np.nan,
                #     rqpe_hail.r_z['Rainfall [mm/h]'])
                # Correct all other variables
                [setattr(
                    rqpe, rp, {(k1): (np.where(
                        (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
                        & (resattc.vars[zh4r] >= thr_zhail),
                        rqpe_hail.r_z['Rainfall [mm/h]'],
                        getattr(rqpe, rp)['Rainfall [mm/h]'])
                        if 'Rainfall' in k1 else v1)
                        for k1, v1 in getattr(rqpe, rp).items()})
                    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]
                if PLOT_METHODS:
                    tp.datavis.rad_display.plot_ppi(
                        resattc.georef, resattc.params, rqpe.r_zopt,
                        xlims=xlims, ylims=ylims, cpy_feats={'status': True},
                        data_proj=ccrs.UTM(zone=32), proj_suffix='utm',
                        fig_size=fig_size)
                for rp in rprods:
                    if frad[0] == fhrad[0][0]:
                        rqpe_acch[cnt][rp] = np.zeros(
                            (resattc.params['nrays'],
                             resattc.params['ngates']))
                    if not np.isscalar(rqpe_acch[cnt][rp]):
                        rqpe_acch[cnt][rp] = np.nansum(
                            (rqpe_acch[cnt][rp],
                             getattr(rqpe, rp)['Rainfall [mm/h]']), axis=0)
                rqpe_dt.append(rqpe.scandatetime)

# %%
# =============================================================================
# Computes accumulations
# =============================================================================
tz = 'Europe/Berlin'
rqpe_acc = tp.qpe.qpe_algs.RadarQPE(resattc)
rqpe_acc.georef = resattc.georef
rqpe_acc.params = resattc.params

rqpe_acch = [{rp: rv / ds_accumtg for rp, rv in rh.items()}
             for rh in rqpe_acch]
rqpe_acc_ed = {rp: np.nansum([i[rp] for i in rqpe_acch
                              if not np.isscalar(i[rp])], axis=0)
               for rp in rprods}

[setattr(rqpe_acc, rp, {'Rainfall [mm]': rqpe_acc_ed[rp]})
 for rp in rprods]

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
# Get rg locations within radar coverage
# =============================================================================
err_bxp = [161, 723, 3969, 14020, 14078, 14081, 14099, 15958, 17593, 19861]
err_jxp = []
err_ess = [1303]
err_fle = ([194, 5297, 3098]
           if START_TIME == dt.datetime(2021, 7, 13, 0, 0)
           else [3098, 4127, 5619, 6061, 6313, 7499, 13713, 14162]
           if START_TIME == dt.datetime(2021, 7, 14, 0, 0)
           else [])
# err_fle = [194, 5297]  # add to 20210713
# 3098, 4127, 5619, 6061, 6313, 7499, 13713, 14162
err_neu = [3, 2473, 14151, 15000]
err_off = []
crrp = [2667, 4741]
# delif = crrp + err_fle
delif = crrp + (err_ess if RSITE == 'Essen'
                else err_fle if RSITE == 'Flechtdorf'
                else [])
rg_data.get_stns_rad(rqpe_acc.georef, rqpe_acc.params, rg_data.dwd_stn_mdata,
                     iradbins=9, dmax2radbin=1,
                     dmax2rad=(100 if rband == 'C' else 100
                               if RSITE == 'Juxpol' else 75),
                     del_by_station_id=delif, plot_methods=False)

# =============================================================================
# Get rg locations within bounding box
# =============================================================================
# rg_data.get_stns_box(rg_data.dwd_stn_mdata, bbox_xlims=(9.076, 9.079),
#                       bbox_ylims=(51.343, 51.346), plot_methods=PLOT_METHODS)

# =============================================================================
# Read DWD rg data
# =============================================================================
rg_data.get_rgdata(STOP_TIME.replace(tzinfo=ZoneInfo(tz)),
                   # rqpe_acc.scandatetime,
                   ds_ncdir=RG_NCDATA, drop_nan=True, drop_thrb=0.1,
                   ds_tres=dt.timedelta(hours=1), plot_methods=False,
                   dt_bkwd=dt.timedelta(hours=EVNTD_HRS+1),
                   ds_accum=dt.timedelta(hours=EVNTD_HRS+1),
                   rprod_fltr=rqpe_acc_ed['r_zopt'], rprod_thr=1,
                   sort_rgdata={'sort': True})

rg_acprecip = {'grid_wgs84x': rg_data.ds_precip['longitude [dd]'],
               'grid_wgs84y': rg_data.ds_precip['latitude [dd]'],
               'Rainfall [mm]': rg_data.ds_precip['rain_sum'].flatten(),
               'altitude [m]': rg_data.ds_precip['altitude [m]'].flatten(),
               'd2rad [km]': rg_data.ds_precip['distance2rad [km]'].flatten()}
# %%
# =============================================================================
# RAIN GAUGE HACCUMS VS RQPE HACCUMS
# =============================================================================
id_largestacc = True
if id_largestacc:
    id_rgs = rg_data.ds_precip['station_id'][:4].flatten()
    # id_rgs = rg_data.ds_precip['station_id'][12:16].flatten()
    # id_rgs = [3499, 1300, 4488, 2947]
    rgsffx = 'L'
else:
    id_rgs = rg_data.ds_precip['station_id'][-4:].flatten()
    # id_rgs = rg_data.ds_precip['station_id'][-6:-2].flatten()
    # id_rgs = [3700, 1424, 7341, 917]
    rgsffx = 'S'

# plt.style.use('default')
# plt.style.use('seaborn-v0_8-darkgrid')

['Solarize_Light2', '_classic_test_patch', '_mpl-gallery',
 '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast',
 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8',
 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark',
 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep',
 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk',
 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid',
 'tableau-colorblind10']

# PLOT_FIGS = True

if PLOT_FIGS:
    vidx_id = np.argwhere(rg_data.ds_precip['station_id'] == id_rgs)[:, 0]
    rg_precip = {k: [dwdr for c, dwdr in enumerate(v) if c in vidx_id]
                 for k, v in rg_data.ds_precip.items()}
    # rg_precip['rain_idt'] = [np.array([[iidt.replace(tzinfo=ZoneInfo(tz))
    #                                     for iidt in idt]
    #                                    for idt in nar])
    #                          for nar in rg_precip['rain_idt']]
    eval_rqp1 = [{k: np.array(
        [np.nanmean(i)
         for i in restimator.flatten()[rg_precip['kd_rbin_idx']]])
        for k, restimator in r1.items()
        if not np.isscalar(restimator)
        } for r1 in rqpe_acch]
    eval_rqp3 = {key: [] for key in rprods}
    for i in eval_rqp1:
        if i:
            for re in rprods:
                eval_rqp3[re].append(i[re])
        else:
            for re in rprods:
                eval_rqp3[re].append(np.zeros_like(id_rgs))
    eval_rqp4 = {k: np.vstack(rq).T for k, rq in eval_rqp3.items()}
    eval_rqp5 = {k: [np.nancumsum(ri) for ri in rq]
                 for k, rq in eval_rqp4.items()}
    eval_rqp6 = {k: np.insert(v, 0, 0, axis=1) for k, v in eval_rqp5.items()}
    # if plot_methods:
    maxplt = 4
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
    mrks = mpl.lines.Line2D.markers
    mrks = [m1 for m1 in mrks.keys() if type(m1) is str]
    mrks = {rp: mrks[cnt] for cnt, rp in enumerate(rprods)}
    for nplot in nplots:
        fig = plt.figure(figsize=(19.2, 11.4))
        fig.suptitle('Cumulative Precipitation -- Rain gauge and '
                     + f'Radar QPE [{RSITE}]', fontsize=18)
        grid = ImageGrid(fig, 111, aspect=False,
                         nrows_ncols=(nrows, ncols),  # label_mode="L",
                         share_all=False, axes_pad=0.5)
        wxsid2plot = [i for i in rg_precip['station_id'][nplot[0]:nplot[-1]]]
        wxsdt2plot = [i for i in rg_precip['rain_idt'][nplot[0]:nplot[-1]]]
        wxsrc2plot = [i for i in rg_precip['rain_cumsum'][nplot[0]:nplot[-1]]]
        w1max = np.nanmax([np.nanmax(i) for i in wxsrc2plot])
        plimit_min = (-1.5 if w1max < 50 else -2.5 if 50 < w1max < 150
                      else -5 if w1max > 150 else -1.5)
        plimit_max = w1max*1.25 + (25 - w1max*1.25) % 25
        for rp in rprods:
            if rp in RPRODSLTX:
                rprodkltx = RPRODSLTX.get(rp)
            else:
                rprodkltx = rp
            for (axg, rgid, rgdt, rgcs, rp1) in zip(
                    grid, wxsid2plot, wxsdt2plot, wxsrc2plot,
                    [i for i in eval_rqp6[rp]]):
                axg.set_title(f'Station: {rgid}', fontsize=16)
                if rp == rprods[0]:
                    axg.plot(np.array(rgdt).flatten(),
                             np.array(rgcs).flatten(), '-', c='k',
                             label='Rain Gauge')
                axg.plot(np.array(rgdt).flatten(), np.array(rp1).flatten(),
                         f'{mrks[rp]}--', label=f'{rprodkltx}')
                axg.xaxis.set_major_locator(locator)
                axg.xaxis.set_major_formatter(formatter)
                axg.set_xlabel('Date and time', fontsize=16)
                axg.set_ylabel('Rainfall [mm]', fontsize=16)
                # axg.set_xlim(htixlim)
                # axg.set_ylim([-1.5, 30.])
                axg.set_ylim([plimit_min, plimit_max])
                # axg.set_ylim([-5, 190.])
                axg.grid(True)
                axg.set_axisbelow(True)
                axg.tick_params(axis='both', which='major', labelsize=14)
                axg.xaxis.get_offset_text().set_size(14)
                # if rgid == idwxst2plot[0]:
                #     axg.spines['top'].set_visible(True)
                #     axg.spines['bottom'].set_visible(False)
                #     axg.spines['right'].set_visible(False)
                # elif rgid == idwxst2plot[1]:
                #     axg.spines['top'].set_visible(True)
                #     axg.spines['bottom'].set_visible(False)
                #     axg.spines['left'].set_visible(False)
                # elif rgid == idwxst2plot[2]:
                #     axg.spines['top'].set_visible(False)
                #     axg.spines['right'].set_visible(False)
                # elif rgid == idwxst2plot[3]:
                #     axg.spines['top'].set_visible(False)
                #     axg.spines['left'].set_visible(False)
            nitems = len(rg_precip['station_id'])
        plt.tight_layout()
        # Use these lines to change the order of the legend
        # (left-to-right instead of top-to-bottom)
        br = lambda n: (sum((nc1[i::n]for i in range(n)), [])
                        for nc1 in axg.get_legend_handles_labels())
        handles, labels = br((len(rprods)//2)+1)
        # handles, labels = axg.get_legend_handles_labels()
        fig.legend(handles, labels, ncols=(len(rprods)//2)+1, shadow=True,
                   columnspacing=1., fontsize=13, loc='upper center',
                   bbox_to_anchor=(0.5, 0.965))
        fig.subplots_adjust(top=0.86)
        plt.show()
    if SAVE_FIGS:
        fname = (f"{START_TIME.strftime('%Y%m%d')}"
                 + f"_{RPARAMS[0]['site_name'][:3].lower()}"
                 + f"_daily_evalrgrqpeh{rgsffx}{appxf}.png")
        plt.savefig(RES_DIR+fname, dpi=200, format='png')

# %%
# =============================================================================
# QPE PRODUCTS
# =============================================================================
rprodsplot = rprods
# rprodsplot = ['r_ah', 'r_aho_kdpo', 'r_kdpopt', 'r_z', 'r_zopt']
# rprodsplot = ['r_zopt']

if PLOT_FIGS:
    for re in sorted(rprodsplot):
        if re in RPRODSLTX:
            rprodkltx = RPRODSLTX.get(re)
        else:
            rprodkltx = re
        bnd = {}
        bnd['[mm]'] = np.array((0.1, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35,
                                40, 45, 50, 55, 60, 70, 80, 90, 100, 115, 130,
                                145, 160, 175, 200))
        unorm = {}
        unorm['[mm]'] = mpc.BoundaryNorm(
            bnd['[mm]'], mpl.colormaps['tpylsc_rad_rainrt'].N, extend='max')
        rqpe_acc1 = {}
        rqpe_acc1 = getattr(rqpe_acc, re)
        rad_display.plot_ppi(rqpe_acc.georef, rqpe_acc.params, rqpe_acc1,
                             cpy_feats={'status': True}, fig_size=fig_size,
                             xlims=xlims, ylims=ylims, points2plot=rg_acprecip,
                             ptsvar2plot='Rainfall [mm]',
                             data_proj=ccrs.PlateCarree(), proj_suffix='wgs84',
                             # unorm=unorm,  # ucmap='turbo',
                             fig_title=(
                                  f'{RSITE} -- {rprodkltx}: '
                                  + f"{rqpe_acc.params['datetime']:%Y-%m-%d}"))
        if SAVE_FIGS:
            fname = (f"{START_TIME.strftime('%Y%m%d')}"
                     + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_"
                     + f"{re.replace('_', '')}{appxf}.png")
            plt.savefig(RES_DIR+fname, dpi=200, format='png')
            plt.close()

# %%
# =============================================================================
# QPE STATS
# =============================================================================
# cmaph = mpl.colormaps['gist_earth_r']
# cmaph = mpl.colormaps['Spectral_r']
# lpv = {'Altitude [m]':
#        [round(np.nanmin(rg_data.ds_precip['altitude [m]']), 2),
#         round(np.nanmax(rg_data.ds_precip['altitude [m]']), -2), 25]}
lpv = {'Altitude [m]': [0, 1000, 11],
       'Distance [km]': [0, 100, 11]}
bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
       for key, value in lpv.items()}
dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
    value,
    mpl.colormaps['gist_earth_r'].N,
    extend='max')
         for key, value in bnd.items()}

if 'bDistance [km]' in bnd.keys():
    dnorm['nDistance [km]'] = mpc.BoundaryNorm(
        bnd['bDistance [km]'], mpl.colormaps['Spectral_r'].N,
        extend='max')
    cmaph = mpl.colormaps['Spectral_r']

rg_data.ds_precip['eval_rqpe'] = {
    k: np.array([np.nanmean(i) for i in restimator.flatten()
                 [rg_data.ds_precip['kd_rbin_idx']]])
    for k, restimator in rqpe_acc_ed.items()}

rg_data.ds_precip['eval_rg'] = {
    k: rg_data.ds_precip['rain_sum'].flatten() for k in rqpe_acc_ed.keys()}

for k, v in rg_data.ds_precip['eval_rg'].items():
    # v[v <= 0.1] = np.nan
    v[np.isnan(rg_data.ds_precip['eval_rg'][k])] = np.nan
    v[rg_data.ds_precip['eval_rqpe'][k] <= 0.1] = np.nan

qpe_stats = {k: tpx.mstats(v, rg_data.ds_precip['eval_rg'][k])
             for k, v in rg_data.ds_precip['eval_rqpe'].items()}

maxplt = 18
nitems = len(rg_data.ds_precip['eval_rqpe'])
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

for nplot in nplots:
    fig = plt.figure(figsize=(19.2, 11))
    fig.suptitle(f'Daily accumulated radar QPE [{RSITE}] '
                 'vs Rain-gauge measured rain totals')
    grid = ImageGrid(fig, 111, aspect=False,
                     nrows_ncols=(nrows, ncols), label_mode='L',
                     # label_mode='keep',
                     share_all=True, axes_pad=0.5,  cbar_location="right",
                     cbar_mode="single", cbar_size="4%", cbar_pad=0.75)
    for (axg, rkeys) in zip(
            grid, [k for k in sorted(rg_data.ds_precip['eval_rqpe'].keys())]):
        if rkeys in RPRODSLTX:
            rprodkltx = RPRODSLTX.get(rkeys)
        else:
            rprodkltx = rkeys
        axg.set_title(f'Rainfall estimator: {rprodkltx}')
        f1 = axg.scatter(rg_data.ds_precip['eval_rg'][rkeys],
                         rg_data.ds_precip['eval_rqpe'][rkeys], edgecolors='k',
                         # c=[rg_acprecip['altitude [m]']], edgecolors='k',
                         c=[rg_data.ds_precip['distance2rad [km]'].flatten()],
                         marker='o', cmap=cmaph, norm=dnorm['nDistance [km]'])
        f2 = axg.scatter(
            0, 0, marker='',
            label=(
                f"n={qpe_stats[rkeys]['N']}"
                + f"\nMAE={qpe_stats[rkeys]['MAE']:2.2f}"
                + f"\nRMSE={qpe_stats[rkeys]['RMSE']:2.2f}"
                # + f"\nNRMSE [%]={qpe_stats[rkeys]['NRMSE [%]']:2.2f}"
                + f"\nNMB [%]={qpe_stats[rkeys]['NMB [%]']:2.2f}"
                + f"\nr={qpe_stats[rkeys]['R_Pearson [-]'][0,1]:.2f}"))
        axg.axline((1, 1), slope=1, c='gray', ls='--')
        axg.set_xlabel('Gauge-measured R [mm]', fontsize=12)
        axg.set_ylabel('Radar rainfall R [mm]', fontsize=12)
        if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
            axg.set_xlim([0, 85])
            axg.set_ylim([0, 85])
        else:
            axg.set_xlim([0, 180])
            axg.set_ylim([0, 180])
            # axg.set_xlim([0, 35])
            # axg.set_ylim([0, 35])
        axg.grid(True)
        axg.legend(loc=2, fontsize=10, handlelength=0, handletextpad=0,
                   fancybox=True)
        plt.show()
    # nitems = len(eval_rqp)
    axg.cax.colorbar(f1)
    axg.cax.tick_params(direction='in', which='both', labelsize=12)
    # axg.cax.toggle_label(True)
    axg.cax.set_title('Distance \n to radar [km]', fontsize=12)
    plt.tight_layout()

if SAVE_FIGS:
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}"
             + f"_daily_qpestats{appxf}.png")
    plt.savefig(RES_DIR+fname, dpi=200, format='png')

# %%
# =============================================================================
# QC indicators
# =============================================================================

converter = mdates.ConciseDateConverter()
mpl.units.registry[np.datetime64] = converter
mpl.units.registry[dt.date] = converter
mpl.units.registry[dt.datetime] = converter

htixlim = [(START_TIME-dt.timedelta(minutes=30)).replace(tzinfo=ZoneInfo(tz)),
           (STOP_TIME+dt.timedelta(minutes=30)).replace(tzinfo=ZoneInfo(tz))]

fig, axs = plt.subplots(4, figsize=(19.2, 11), sharex=True)
fig.suptitle(f'{RSITE} - QC indicators and metrics')
# MLYr height
ax = axs[0]
ax.plot(rqpe_dt, np.array(mlyrh)[:, 0], label='ML_TOP')
# ax.plot(rqpe_dt, np.array(mlyrh)[:, 1], label='ML_THICKNESS')
ax.plot(rqpe_dt, np.array(mlyrh)[:, 2], label='ML_BOTTOM')
ax.fill_between(rqpe_dt, np.array(mlyrh)[:, 0],
                np.array(mlyrh)[:, 0] - np.array(mlyrh)[:, 1], alpha=0.4)
ax.set_ylabel('Height [km]')
ax.legend(loc=2)
ax.grid()
# minSNR
ax = axs[1]
ax.plot(rqpe_dt, np.array(r_minsnr))
ax.plot(rqpe_dt, np.nanmean(r_minsnr) * np.ones_like(r_minsnr),
        c='dodgerblue', ls=':',
        label=fr'$\overline{{minSNR}}$ = {np.nanmean(r_minsnr):0.2f}')
ax.legend(loc=2)
# ax.set_ylim((-26, -24))
# ax.set_ylim((-20.5, -18.5))
# ax.set_ylim((39, 42))
ax.set_ylabel('min_SNR [dB]')
ax.grid()
# ZDR0
ax = axs[2]
ax.plot(rqpe_dt, np.array(rzdr0), label=r'$Z_{DR}(O)$')
ax.plot(rqpe_dt, np.nanmean(rzdr0) * np.ones_like(rzdr0), c='dodgerblue',
        ls=':', label=r'$\overline{{Z_{{DR}}(0)}}$ = '
        + f'{np.nanmean(rzdr0):0.2f}')
ax.legend(loc=2)
ax.set_ylabel(r'$Z_{DR}(O)$ [dB]')
ax.grid()
# PhiDP0
ax = axs[3]
ax.plot(rqpe_dt, np.array(rphidp0))
ax.plot(rqpe_dt, np.nanmean(rphidp0) * np.ones_like(rphidp0), c='dodgerblue',
        ls=':', label=r'$\overline{{\Phi_{{DP}}(0)}}$ = '
        + f'{np.nanmean(rphidp0):0.2f}')
ax.legend(loc=2)
ax.set_ylabel(r'$\Phi_{DP}(0)$ [deg]')
ax.grid()
ax.set_xlabel('Date and time')
ax.set_xlim(htixlim)
plt.tight_layout()

if SAVE_FIGS:
    RES_DIR2 = RES_DIR.replace('rsite_qpe', 'rsite_dailyqc')
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_qci.png")
    plt.savefig(RES_DIR2 + fname, format='png')

fig, axs = plt.subplots(2, figsize=(19.2, 5.5), sharex=True)
fig.suptitle(f'{RSITE} - $R(Z_H)$, $R(K_{{DP}})$ -- Optimised coefficients')

# ZOpt coeffs
ax = axs[0]
c1 = 'darkorange'
ax.plot(rqpe_dt, [a[0] for a in radapt_coeffs['z_opt']],
        label='a', ls='-', c=c1)
ax.tick_params(axis='y', labelcolor=c1)
ax.set_ylabel(r'$Z=aR^{b}$')
# ax.set_ylim((0, 500))
ax.grid()
c2 = 'dodgerblue'
ax2 = ax.twinx()
ax2.plot(rqpe_dt, [a[1] for a in radapt_coeffs['z_opt']], c=c2,
         label='b', ls=':')
ax2.tick_params(axis='y', labelcolor=c2)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2)
# KDPOpt coeffs
ax = axs[1]
ax.plot(rqpe_dt, [a[0] for a in radapt_coeffs['kdp_opt']],
        label='a', ls='-', c=c1)
ax.tick_params(axis='y', labelcolor=c1)
ax.set_ylabel(r'$R=aK_{DP}^{b}$')
# ax.set_ylim((0, 35))
ax.grid()
ax2 = ax.twinx()
ax2.plot(rqpe_dt, [a[1] for a in radapt_coeffs['kdp_opt']], c=c2,
         label='b', ls=':')
ax2.tick_params(axis='y', labelcolor=c2)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2)
ax.set_xlabel('Date and time')
ax.set_xlim(htixlim)
plt.tight_layout()

if SAVE_FIGS:
    # RES_DIR2 = RES_DIR.replace('rsite_qpe', 'rsite_dailyqc')
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_coeffs.png")
    plt.savefig(RES_DIR + fname, dpi=200, format='png')

if SAVE_COEFFS:
    # RES_DIR2 = RES_DIR.replace('rsite_qpe', 'rsite_dailyqc')
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_ropt.tpy")
    frstats = {'dt': rqpe_dt,
               'rz_opt': radapt_coeffs['z_opt'],
               'rkdp_opt': radapt_coeffs['kdp_opt']}
    with open(RES_DIR+fname, 'wb') as f:
        pickle.dump(frstats, f, pickle.HIGHEST_PROTOCOL)
