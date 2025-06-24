#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:20:10 2025

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

# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24 hr [NO BXP]
START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24 hr [NO BXP]

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

for cnt, fhrad in enumerate(tqdm(ds_fullgidx,
                                 desc=f'Computing RQPE [{RSITE}]')):
    for frad in fhrad:
        if frad[1] is not np.nan:
            # =============================================================================
            # Import data from wradlib to towerpy
            # =============================================================================
            rpolara = tpx.Rad_scan(frad[1], f'{RSITE}')
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
                zh4rkdpo = 'ZH+ [dBZ]'  # ZH(ATTC)
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
                    mlyr=rmlyr, rband=rband, kdpmed=0.5,
                    zh_thr=((40, 50) if rband == 'X' else (44.5, 45.5)),
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
                    maxpia=(50 if rband == 'X' else 50),
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
            for rp in rprods:
                if frad[0] == fhrad[0][0]:
                    rqpe_acch[cnt][rp] = np.zeros(
                        (rpolara.params['nrays'],
                         rpolara.params['ngates']))
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
rqpe_acc = tp.qpe.qpe_algs.RadarQPE(rpolara)
rqpe_acc.georef = rpolara.georef
rqpe_acc.params = rpolara.params

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
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'

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
err_fle = [3098, 4127, 5619, 6061, 6313, 7499, 13713, 14162]
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
# rg_data.get_stns_box(rg_data.dwd_stn_mdata, bbox_xlims=(7.2, 7.45),
#                       bbox_ylims=(50.15, 50.45), plot_methods=PLOT_METHODS)

# =============================================================================
# Read DWD rg data
# =============================================================================
rg_data.get_rgdata(STOP_TIME.replace(tzinfo=ZoneInfo(tz)),
                   # rqpe_acc.scandatetime,
                   ds_ncdir=RG_NCDATA, drop_nan=True, drop_thrb=0.1,
                   ds_tres=dt.timedelta(hours=1), plot_methods=False,
                   dt_bkwd=dt.timedelta(hours=EVNTD_HRS+1),
                   ds_accum=dt.timedelta(hours=EVNTD_HRS+1),
                   rprod_fltr=rqpe_acc_ed['r_zopt'], rprod_thr=0.1,
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

if PLOT_FIGS:
    vidx_id = np.argwhere(rg_data.ds_precip['station_id'] == id_rgs)[:, 0]
    rg_precip = {k: [dwdr for c, dwdr in enumerate(v) if c in vidx_id]
                 for k, v in rg_data.ds_precip.items()}
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
        fig.suptitle(f"RG stations -- {RSITE} [Cumulative Precipitation]")
        grid = ImageGrid(fig, 111, aspect=False,
                         nrows_ncols=(nrows, ncols),  # label_mode="L",
                         share_all=False, axes_pad=0.5)
        for rp in rprods:
            for (axg, rgid, rgdt, rgcs, rp1) in zip(
                    grid,
                    [i for i in rg_precip['station_id'][nplot[0]:nplot[-1]]],
                    [i for i in rg_precip['rain_idt'][nplot[0]:nplot[-1]]],
                    [i for i in rg_precip['rain_cumsum'][nplot[0]:nplot[-1]]],
                    [i for i in eval_rqp6[rp]]):
                axg.set_title(f'Station: {rgid}')
                if rp == rprods[0]:
                    axg.plot(np.array(rgdt).flatten(),
                             np.array(rgcs).flatten(), '-', c='k',
                             label='Rain Gauge')
                axg.plot(np.array(rgdt).flatten(), np.array(rp1).flatten(),
                         f'{mrks[rp]}--', label=f'{rp}')
                axg.xaxis.set_major_locator(locator)
                axg.xaxis.set_major_formatter(formatter)
                axg.set_xlabel('Date and time', fontsize=12)
                axg.set_ylabel('mm', fontsize=12)
                # axg.set_xlim(htixlim)
                axg.grid(True)
                axg.legend(loc=2)
            plt.show()
            nitems = len(rg_precip['station_id'])
        plt.tight_layout()
    if SAVE_FIGS:
        fname = (f"{START_TIME.strftime('%Y%m%d')}"
                 + f"_{RPARAMS[0]['site_name'][:3].lower()}"
                 + f"_daily_rgrqpeh{rgsffx}{appxf}.png")
        plt.savefig(RES_DIR+fname, format='png')
# %%
# =============================================================================
# QPE PRODUCTS
# =============================================================================
rprodsplot = rprods

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
                             unorm=unorm,  # ucmap='turbo',
                             fig_title=(
                                  f'{RSITE} -- {rprodkltx}: '
                                  + f"{rqpe_acc.params['datetime']:%Y-%m-%d}"))
        if SAVE_FIGS:
            fname = (f"{START_TIME.strftime('%Y%m%d')}"
                     + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_"
                     + f"{re.replace('_', '')}{appxf}.png")
            plt.savefig(RES_DIR+fname, format='png')
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
            # axg.set_xlim([0, 35])
            # axg.set_ylim([0, 35])
            axg.set_xlim([0, 180])
            axg.set_ylim([0, 180])
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
             + f"_daily_rqpestats{appxf}.png")
    plt.savefig(RES_DIR+fname, format='png')

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

fig, axs = plt.subplots(6, figsize=(19.2, 11), sharex=True)
fig.suptitle(f'{RSITE} - QC indicators and metrics')
# MLYr height
ax = axs[0]
# ax.plot(rqpe_dt, np.array(mlyrh)[:, 0], label='ML_TOP')
# # ax.plot(rqpe_dt, np.array(mlyrh)[:, 1], label='ML_THICKNESS')
# ax.plot(rqpe_dt, np.array(mlyrh)[:, 2], label='ML_BOTTOM')
# ax.fill_between(rqpe_dt, np.array(mlyrh)[:, 0],
#                 np.array(mlyrh)[:, 0] - np.array(mlyrh)[:, 1], alpha=0.4)
ax.set_ylabel('Height [km]')
ax.legend(loc=2)
ax.grid()
# minSNR
ax = axs[1]
# ax.plot(rqpe_dt, np.array(r_minsnr))
# ax.plot(rqpe_dt, np.nanmean(r_minsnr) * np.ones_like(r_minsnr),
#         c='dodgerblue', ls=':',
#         label=fr'$\overline{{minSNR}}$ = {np.nanmean(r_minsnr):0.2f}')
ax.legend(loc=2)
# ax.set_ylim((-26, -24))
# ax.set_ylim((-20.5, -18.5))
# ax.set_ylim((39, 42))
ax.set_ylabel('min_SNR [dB]')
ax.grid()
# ZDR0
ax = axs[2]
# ax.plot(rqpe_dt, np.array(rzdr0), label=r'$Z_{DR}(O)$')
ax.legend(loc=2)
ax.set_ylabel(r'$Z_{DR}(O)$ [dB]')
ax.grid()
# PhiDP0
ax = axs[3]
# ax.plot(rqpe_dt, np.array(rphidp0))
# ax.plot(rqpe_dt, np.nanmean(rphidp0) * np.ones_like(rphidp0), c='dodgerblue',
#         ls=':', label=r'$\overline{{\Phi_{{DP}}(0)}}$ = '
#         + f'{np.nanmean(rphidp0):0.2f}')
ax.legend(loc=2)
ax.set_ylabel(r'$\Phi_{DP}(0)$ [deg]')
ax.grid()
# ZOpt coeffs
ax = axs[4]
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
ax = axs[5]
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
    RES_DIR2 = RES_DIR.replace('rsite_qpe', 'rsite_dailyqc')
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_qci.png")
    plt.savefig(RES_DIR2 + fname, format='png')

if SAVE_COEFFS:
    RES_DIR2 = RES_DIR.replace('rsite_qpe', 'rsite_dailyqc')
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_ropt.tpy")
    frstats = {'dt': rqpe_dt,
               'rz_opt': radapt_coeffs['z_opt'],
               'rkdp_opt': radapt_coeffs['kdp_opt']}
    with open(RES_DIR2+fname, 'wb') as f:
        pickle.dump(frstats, f, pickle.HIGHEST_PROTOCOL)


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
    # rpolara.vars,
    rmlyr.mlyr_limits,
    # var2plot='ML [-]',
    # var2plot='rhoHV [-]',
    # var2plot='KDP [deg/km]',
    # var2plot='AH [dB/km]',
    # var2plot='Rainfall [mm/h]',
    # var2plot='PhiDP [deg]',
    # var2plot='ZDR [dB]',
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

