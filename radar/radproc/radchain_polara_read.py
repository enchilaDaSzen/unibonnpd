#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:20:10 2025

@author: dsanchez
"""

import datetime as dt
# from zoneinfo import ZoneInfo
# import os
# import pickle
import numpy as np
import towerpy as tp
from itertools import zip_longest
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.colors as mpc
# from mpl_toolkits.axes_grid1 import ImageGrid
# import matplotlib.dates as mdates
import cartopy.crs as ccrs
# from towerpy.datavis import rad_display
from radar import twpext as tpx
from radar.rparams_dwdxpol import RPARAMS, RPRODSLTX
# from tqdm import tqdm

# =============================================================================
# Define working directory, and date-time
# =============================================================================

START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24 hr [NO BXP]

# STOP_TIME = dt.datetime(2021, 7, 13, 23, 59)
# EVNTD_HRS = round((STOP_TIME - START_TIME).total_seconds() / 3600)
EVNTD_HRS = 24
STOP_TIME = START_TIME + dt.timedelta(hours=EVNTD_HRS)

QPE_TRES = dt.timedelta(minutes=5)

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = ("/automount/realpep/upload/makter/exportNew/"
         + f"{START_TIME.strftime('%Y%m')}/")

# =============================================================================
# Define radar site
# =============================================================================
# Choose only one site at a time
# Essen, Flechtdorf, Neuheilenbach, Offenthal
RSITE = 'Offenthal'
RPARAMS = [next(item for item in RPARAMS if item['site_name'] == RSITE)]

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

RES_DIR = (LWDIR + f"pd_rdres/qpe_{START_TIME.strftime('%Y%m%d')}/rsite_qpe_"
           + 'polara/')

if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
    RPRODSLTX['r_kdpo'] = '$R(K_{DP})[OV]$'
    RPRODSLTX['r_zo'] = '$R(Z_{H})[OA]$'
    RPRODSLTX['r_aho_kdpo'] = '$R(A_{H}, K_{DP})[OV]$'
# %%
# =============================================================================
# List POLARA radar data
# =============================================================================
RDQC_FILES = {RSITE: tpx.get_listfilesdwd(RSITE, START_TIME, STOP_TIME,
                                          working_dir=EWDIR)}

# %%
# Essen, Flechtdorf, Neuheilenbach, Offenthal
if RSITE == 'Essen':
    kwsffx = 'ess'
elif RSITE == 'Flechtdorf':
    kwsffx = 'fld'
elif RSITE == 'Neuheilenbach':
    kwsffx = 'nhb'
elif RSITE == 'Offenthal':
    kwsffx = 'oft'
rs_ts = {k1: np.array([dt.datetime.strptime(v2[0][v2[0].find('00-')+3:
                                               v2[0].find(f'{kwsffx}')-1],
                                            '%Y%m%d%H%M%S%f')
                       for v2 in v1]) for k1, v1 in RDQC_FILES.items()}
rs_fts = {k1: tpx.fill_timeseries(rs_ts[k1],
                                  range(len(rs_ts[k1])),
                                  stspdt=(START_TIME, STOP_TIME),
                                  toldt=dt.timedelta(minutes=2))[1]
          for k1, v1 in RDQC_FILES.items()}

RDQC_FILES = {k1: [RDQC_FILES[k1][i] if ~np.isnan(i)
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

# %%
# =============================================================================
# QPE using POLARA PPI rdata
# =============================================================================
rqpe_dt = []
mlyrh = []
r_minsnr = []
rphidp0 = []
rzdr0 = []
radapt_coeffs = {'z_opt': [], 'kdp_opt': []}
rqpe_acch = [{rp: 0 for rp in rprods} for h1 in ds_fullgidx]
vld1 = [cnt for cnt, i1 in enumerate(RDQC_FILES) if type(i1) is list][0]

# frad = RDQC_FILES[210]
# frad = RDQC_FILES[122]

# =============================================================================
# Import data from wradlib to towerpy
# =============================================================================
rpolara = tpx.Rad_scan(RDQC_FILES[174], f'{RSITE}')
rpolara.ppi_dwd_polara(get_rvar='all')
# ============================================================================
# Melting layer allocation
# ============================================================================
rpolara.vars['ML [-]'][rpolara.vars['ML [-]'] == -1] = 3
rpolara.vars['ML [-]'][rpolara.vars['ML [-]'] < 0.5] = 2
mlh_idx = [np.nonzero(np.diff(nray))[0]
           if np.count_nonzero(np.diff(nray)) > 1
           else np.array([np.nonzero(np.diff(nray))[0][0],
                          rpolara.params['ngates']-1])
           for nray in rpolara.vars['ML [-]']]
mlh = [np.array([nbh[mlh_idx[c1][0]], nbh[mlh_idx[c1][1]]])
       for c1, nbh in enumerate(rpolara.georef['beam_height [km]'])]
rmlyr = tp.ml.mlyr.MeltingLayer(rpolara)
rmlyr.ml_top = np.array([i[1] for i in mlh])
rmlyr.ml_bottom = np.array([i[0] for i in mlh])
rmlyr.ml_thickness = np.array([i[1] - i[0] for i in mlh])
rmlyr.ml_ppidelimitation(rpolara.georef, rpolara.params,
                         rpolara.vars, plot_method=PLOT_METHODS)
rpolara.vars.pop('ML [-]')
mlyrh.append([rmlyr.ml_top, rmlyr.ml_thickness,
              rmlyr.ml_bottom])
# =============================================================================
# Partial beam blockage correction
# =============================================================================
rzhah = tp.attc.r_att_refl.Attn_Refl_Relation(rpolara)
rzhah.ah_zh(rpolara.vars, zh_upper_lim=55, temp=temp, rband='C',
            copy_ofr=True, data2correct=rpolara.vars)
mov_avrgf_len = (1, 3)
zh_difnan = np.where(rzhah.vars['diff [dBZ]'] == 0, np.nan,
                     rzhah.vars['diff [dBZ]'])
zhpdiff = np.array([np.nanmedian(i) if ~np.isnan(np.nanmedian(i))
                    else 0 for cnt, i in enumerate(zh_difnan)])
zhpdiff_pad = np.pad(zhpdiff, mov_avrgf_len[1]//2, mode='wrap')
zhplus_maf = np.ma.convolve(
    zhpdiff_pad, np.ones(mov_avrgf_len[1])/mov_avrgf_len[1],
    mode='valid')
rpolara.vars['ZH+ [dBZ]'] = np.array(
    [rpolara.vars['ZH [dBZ]'][cnt] - i if i == 0
     else rpolara.vars['ZH [dBZ]'][cnt] - zhplus_maf[cnt]
     for cnt, i in enumerate(zhpdiff)])
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rpolara.georef, rpolara.params, rpolara.vars, rpolara.vars,
        var2plot1='ZH [dBZ]', var2plot2='ZH+ [dBZ]')
rband = next(item['rband'] for item in RPARAMS
             if item['site_name'] == rpolara.site_name)
# =============================================================================
# KDP Derivation
# =============================================================================
# Remove negative KDP values in rain region and within ZH threshold
zh_kdp = 'ZH+ [dBZ]'
rpolara.vars['KDP [deg/km]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rpolara.vars['KDP [deg/km]'] < 0)
    & (rpolara.vars[zh_kdp] > 5),
    0, rpolara.vars['KDP [deg/km]'])
# TODO: FOR C BAND ZH(ATTC) WORKS BETTER FOR KDP, WHY?
if rband == 'C':
    zh4rkdpo = 'ZH [dBZ]'  # ZH(ATTC)
else:
    zh4rkdpo = 'ZH+ [dBZ]'  # ZH(AH)
zh4rzo = 'ZH+ [dBZ]'  # ZH(AH)
zh4r = 'ZH+ [dBZ]'  # ZH(AH)
zdr4r = 'ZDR [dB]'
ah4r = 'AH [dB/km]'
adpr = 'ADP [dB/km]'
kdp4rkdpo = 'KDP [deg/km]'  # AH
kdp4r = 'KDP [deg/km]'  # Vulpiani+AH
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
rqpe = tp.qpe.qpe_algs.RadarQPE(rpolara)
if 'r_adp' in rprods:
    rqpe.adp_to_r(
        rpolara.vars[adpr], mlyr=rmlyr, temp=temp, rband=rband,
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_ah' in rprods:
    rqpe.ah_to_r(
        rpolara.vars[ah4r], mlyr=rmlyr, temp=temp, rband=rband,
        # a=rah_a, b=rah_b,
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_kdp' in rprods:
    rqpe.kdp_to_r(
        rpolara.vars[kdp4r], mlyr=rmlyr,
        a=next(item['rkdp_a'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        b=next(item['rkdp_b'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_z' in rprods:
    rqpe.z_to_r(rpolara.vars[zh4r], mlyr=rmlyr,
                a=next(item['rz_a'] for item in RPARAMS
                       if item['site_name'] == rqpe.site_name),
                b=next(item['rz_b'] for item in RPARAMS
                       if item['site_name'] == rqpe.site_name),
                beam_height=rpolara.georef['beam_height [km]'])
# =============================================================================
# Hybrid estimators
# =============================================================================
if 'r_kdp_zdr' in rprods:
    rqpe.kdp_zdr_to_r(
        rpolara.vars[kdp4r], rpolara.vars[zdr4r], mlyr=rmlyr,
        a=next(item['rkdpzdr_a'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        b=next(item['rkdpzdr_b'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        c=next(item['rkdpzdr_c'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_z_ah' in rprods:
    rqpe.z_ah_to_r(
        rpolara.vars[zh4r], rpolara.vars[ah4r], mlyr=rmlyr,
        z_thld=z_thld, temp=temp, rband=rband,
        rz_a=next(item['rz_a'] for item in RPARAMS
                  if item['site_name'] == rqpe.site_name),
        rz_b=next(item['rz_b'] for item in RPARAMS
                  if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_z_kdp' in rprods:
    rqpe.z_kdp_to_r(
        rpolara.vars[zh4r], rpolara.vars[kdp4r], mlyr=rmlyr,
        z_thld=z_thld,
        rz_a=next(item['rz_a'] for item in RPARAMS
                  if item['site_name'] == rqpe.site_name),
        rz_b=next(item['rz_b'] for item in RPARAMS
                  if item['site_name'] == rqpe.site_name),
        rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                    if item['site_name'] == rqpe.site_name),
        rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                    if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_z_zdr' in rprods:
    rqpe.z_zdr_to_r(
        rpolara.vars[zh4r], rpolara.vars[zdr4r], mlyr=rmlyr,
        a=next(item['rzhzdr_a'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        b=next(item['rzhzdr_b'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        c=next(item['rzhzdr_c'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
if 'r_ah_kdp' in rprods:
    rqpe.ah_kdp_to_r(
        rpolara.vars[zh4r], rpolara.vars[ah4r],
        rpolara.vars[kdp4r], mlyr=rmlyr, z_thld=z_thld,
        temp=temp, rband=rband,
        rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                    if item['site_name'] == rqpe.site_name),
        rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                    if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
# =============================================================================
# Adaptive estimators
# =============================================================================
rqpe_opt = tp.qpe.qpe_algs.RadarQPE(rpolara)
if 'r_kdpopt' in rprods:
    rkdp_fit = tpx.rkdp_opt(
        rpolara.vars[kdp4rkdpo], rpolara.vars[zh4rkdpo],
        mlyr=rmlyr, rband=rband,
        rkdp_stv=(next(item['rkdp_a'] for item in RPARAMS
                       if item['site_name'] == rqpe.site_name),
                  next(item['rkdp_b'] for item in RPARAMS
                       if item['site_name'] == rqpe.site_name)))
    rqpe_opt.kdp_to_r(
        rpolara.vars[kdp4r], a=rkdp_fit[0], b=rkdp_fit[1],
        mlyr=rmlyr,
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_kdpopt = rqpe_opt.r_kdp
    radapt_coeffs['kdp_opt'].append(
        (((rkdp_fit[0], rkdp_fit[1]))))
if 'r_zopt' in rprods:
    rzh_fit = tpx.rzh_opt(
        rpolara.vars[zh4rzo], rqpe.r_ah, rpolara.vars[ah4r],
        # pia=rpolara.vars['PIA [dB]'], mlyr=rmlyr,
        pia=None, mlyr=rmlyr,
        # maxpia=(20 if rband == 'X' else 10),
        maxpia=(50 if rband == 'X' else 20),
        rzfit_b=(2.14 if rband == 'X' else 1.6),
        rz_stv=[next(item['rz_a'] for item in RPARAMS
                     if item['site_name'] == rqpe.site_name),
                next(item['rz_b'] for item in RPARAMS
                     if item['site_name'] == rqpe.site_name)],
        plot_method=PLOT_METHODS)
    rqpe_opt.z_to_r(
        rpolara.vars[zh4r], mlyr=rmlyr, a=rzh_fit[0],
        b=rzh_fit[1],
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zopt = rqpe_opt.r_z
    radapt_coeffs['z_opt'].append(((
                          (rqpe.r_zopt['coeff_a'],
                           rqpe.r_zopt['coeff_b']))))
if 'r_zopt_ah' in rprods and 'r_zopt' in rprods:
    rqpe_opt.z_ah_to_r(
        rpolara.vars[zh4r], rpolara.vars[ah4r], mlyr=rmlyr,
        z_thld=z_thld, temp=temp, rband=rband,
        rz_a=rzh_fit[0], rz_b=rzh_fit[1],
        # rah_a=67, rah_b=0.78,
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zopt_ah = rqpe_opt.r_z_ah
if 'r_zopt_kdp' in rprods and 'r_zopt' in rprods:
    rqpe_opt.z_kdp_to_r(
        rpolara.vars[zh4r], rpolara.vars[kdp4r], mlyr=rmlyr,
        z_thld=z_thld, rz_a=rzh_fit[0], rz_b=rzh_fit[1],
        rkdp_a=next(item['rkdp_a'] for item in RPARAMS
                    if item['site_name'] == rqpe.site_name),
        rkdp_b=next(item['rkdp_b'] for item in RPARAMS
                    if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zopt_kdp = rqpe_opt.r_z_kdp
rqpe_opt2 = tp.qpe.qpe_algs.RadarQPE(rpolara)
if ('r_zopt_kdpopt' in rprods and 'r_zopt' in rprods
        and 'r_kdpopt' in rprods):
    rqpe_opt2.z_kdp_to_r(
        rpolara.vars[zh4r], rpolara.vars[kdp4r], mlyr=rmlyr,
        z_thld=z_thld, rz_a=rzh_fit[0], rz_b=rzh_fit[1],
        rkdp_a=rkdp_fit[0], rkdp_b=rkdp_fit[1],
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zopt_kdpopt = rqpe_opt2.r_z_kdp
if ('r_ah_kdpopt' in rprods and 'r_kdpopt' in rprods):
    rqpe_opt2.ah_kdp_to_r(
        rpolara.vars[zh4r], rpolara.vars[ah4r],
        rpolara.vars[kdp4r], mlyr=rmlyr, z_thld=z_thld,
        temp=temp, rband=rband,
        # rkdp_a=next(item['rkdp_a'] for item in RPARAMS
        #             if item['site_name'] == rqpe.site_name),
        # rkdp_b=next(item['rkdp_b'] for item in RPARAMS
        #             if item['site_name'] == rqpe.site_name),
        rkdp_a=rkdp_fit[0], rkdp_b=rkdp_fit[1],
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_ah_kdpopt = rqpe_opt2.r_ah_kdp
# =============================================================================
# Estimators using non-fully corrected variables
# =============================================================================
rqpe_nfc = tp.qpe.qpe_algs.RadarQPE(rpolara)
zh4r2 = 'ZH [dBZ]'  # ZHattc
kdp4r2 = 'KDP [deg/km]'  # Vulpiani
if 'r_zo' in rprods:
    rqpe_nfc.z_to_r(
        rpolara.vars[zh4r2], mlyr=rmlyr, a=rz_a, b=rz_b,
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zo = rqpe_nfc.r_z
if 'r_kdpo' in rprods:
    rqpe_nfc.kdp_to_r(
        rpolara.vars[kdp4r2], mlyr=rmlyr, a=rkdp_a, b=rkdp_b,
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_kdpo = rqpe_nfc.r_kdp
if 'r_zo_ah' in rprods:
    rqpe_nfc.z_ah_to_r(
        rpolara.vars[zh4r2], rpolara.vars[ah4r], rz_a=rz_a,
        rz_b=rz_b, rah_a=rah_a, rah_b=rah_b, z_thld=z_thld,
        mlyr=rmlyr, rband=rband,
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zo_ah = rqpe_nfc.r_z_ah
if 'r_zo_kdp' in rprods:
    rqpe_nfc.z_kdp_to_r(
        rpolara.vars[zh4r2], rpolara.vars[kdp4r2], rz_a=rz_a,
        rz_b=rz_b, rkdp_a=rkdp_a, rkdp_b=rkdp_b, z_thld=z_thld,
        mlyr=rmlyr,
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zo_kdp = rqpe_nfc.r_z_kdp
if 'r_zo_zdr' in rprods:
    rqpe_nfc.z_zdr_to_r(
        rpolara.vars[zh4r2], rpolara.vars[zdr4r], mlyr=rmlyr,
        a=next(item['rzhzdr_a'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        b=next(item['rzhzdr_b'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        c=next(item['rzhzdr_c'] for item in RPARAMS
               if item['site_name'] == rqpe.site_name),
        beam_height=rpolara.georef['beam_height [km]'])
    rqpe.r_zo_zdr = rqpe_nfc.r_z_zdr
if 'r_aho_kdpo' in rprods:
    rqpe_nfc.ah_kdp_to_r(
        rpolara.vars[zh4r2], rpolara.vars[ah4r],
        rpolara.vars[kdp4r2], mlyr=rmlyr, temp=temp,
        z_thld=z_thld, rband=rband,
        rah_a=rah_a, rah_b=rah_b, rkdp_a=rkdp_a, rkdp_b=rkdp_b,
        beam_height=rpolara.georef['beam_height [km]'])
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
rqpe_ml = tp.qpe.qpe_algs.RadarQPE(rpolara)
rqpe_ml.z_to_r(rpolara.vars[zh4r],
               a=next(item['rz_a'] for item in RPARAMS
                      if item['site_name'] == rqpe.site_name),
               b=next(item['rz_b'] for item in RPARAMS
                      if item['site_name'] == rqpe.site_name))
rqpe_ml.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 2)
    & (rpolara.vars[zh4r] > thr_zwsnw),
    rqpe_ml.r_z['Rainfall [mm/h]']*f_rz_ml, np.nan)
# additional factor for the RZ relation is applied to data
# above the ML
rqpe_sp = tp.qpe.qpe_algs.RadarQPE(rpolara)
rqpe_sp.z_to_r(rpolara.vars[zh4r],
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
rqpe_hail = tp.qpe.qpe_algs.RadarQPE(rpolara)
rqpe_hail.z_to_r(
    rpolara.vars[zh4r],
    a=next(item['rz_haila'] for item in RPARAMS
           if item['site_name'] == rqpe.site_name),
    b=next(item['rz_hailb'] for item in RPARAMS
           if item['site_name'] == rqpe.site_name))
rqpe_hail.r_z['Rainfall [mm/h]'] = np.where(
    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1)
    & (rpolara.vars[zh4r] >= thr_zhail),
    rqpe_hail.r_z['Rainfall [mm/h]'], 0)
# Set a limit in range
# max_rkm = 270.
# grid_range = (
#     np.ones_like(rpolara.georef['beam_height [km]'])
#     * rpolara.georef['range [m]']/1000)
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
        & (rpolara.vars[zh4r] >= thr_zhail),
        rqpe_hail.r_z['Rainfall [mm/h]'],
        getattr(rqpe, rp)['Rainfall [mm/h]'])
        if 'Rainfall' in k1 else v1)
        for k1, v1 in getattr(rqpe, rp).items()})
    for rp in rqpe.__dict__.keys() if rp.startswith('r_')]
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(
        rpolara.georef, rpolara.params, rqpe.r_zopt,
        xlims=xlims, ylims=ylims, cpy_feats={'status': True},
        data_proj=ccrs.UTM(zone=32), proj_suffix='utm',
        fig_size=fig_size)

# %%
# =============================================================================
# OUTPUT
# =============================================================================
rd_qcatc = tp.attc.attc_zhzdr.AttenuationCorrection(rpolara)
rd_qcatc.georef = rpolara.georef
rd_qcatc.params = rpolara.params
rd_qcatc.vars = dict(rpolara.vars)

# del rd_qcatc.vars['alpha [-]']
# del rd_qcatc.vars['beta [-]']
# # del rd_qcatc.vars['PIA [dB]']
# # del rd_qcatc.vars['PhiDP [deg]']
# del rd_qcatc.vars['PhiDP* [deg]']
# del rd_qcatc.vars['ADP [dB/km]']
# # del rd_qcatc.vars['AH [dB/km]']
# # rd_qcatc.vars['AH [dB/km]'] =
# # rd_qcatc.vars['KDP* [deg/km]'] = rkdpv['KDP [deg/km]']
# # rd_qcatc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']
# rd_qcatc.vars['rhoHV [-]'] = rozdr.vars['rhoHV [-]']
# rd_qcatc.vars['Rainfall [mm/h]'] = rqpe.r_zopt['Rainfall [mm/h]']

# if PLOT_METHODS:
#     tp.datavis.rad_display.plot_setppi(rd_qcatc.georef, rd_qcatc.params,
#                                        rd_qcatc.vars, mlyr=rmlyr)

tp.datavis.rad_display.plot_setppi(rpolara.georef, rpolara.params,
                                   rpolara.vars,
                                   vars_bounds={'AH [dB/km]': [0, .15, 17],
                                                'KDP [deg/km]': [-1, 3, 17],
                                                'ZH+ [dBZ]': [-10, 60, 15]})

tp.datavis.rad_interactive.ppi_base(
    rpolara.georef, rpolara.params,
    rpolara.vars,
    # rmlyr.mlyr_limits,
    # var2plot='ML [-]',
    # var2plot='rhoHV [-]',
    # var2plot='KDP [deg/km]',
    # var2plot='AH [dB/km]',
    # var2plot='Rainfall [mm/h]',
    # var2plot='PhiDP [deg]',
    # var2plot='ZDR [dB]',
    var2plot='ZH+ [dBZ]',
    # proj='polar',
    vars_bounds={'ML [-]': [1, 4, 4],
                 'KDP [deg/km]': (-1, 3, 17),
                 'AH [dB/km]': (0, 0.15, 17)},
    # radial_xlims=(10, 62.5),
    # radial_ylims={'PhiDP [deg]': (-5, 270),}
    # ppi_xlims=[-40, 40], ppi_ylims=[-40, 40]
    mlyr=rmlyr,
    # cbticks=rmlyr.regionID, ucmap='tpylc_div_yw_gy_bu',
    )
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()

