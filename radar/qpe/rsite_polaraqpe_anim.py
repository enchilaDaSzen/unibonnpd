#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:09:09 2024

@author: dsanchez
"""
import datetime as dt
import os
import pickle
import numpy as np
import towerpy as tp
from towerpy.utils.radutilities import find_nearest
from tqdm import tqdm
from itertools import zip_longest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from radar import twpext as tpx
from radar.rparams_dwdxpol import RPARAMS, RPRODSLTX
import matplotlib.animation as animation
# import cartopy.io.img_tiles as cimgt
# from towerpy.utils import unit_conversion as tpuc
# from utils import radutilities as rut
# import rad_display_dev as rdd
# import anim_plot_ppi as animppi
# import anim_base

# =============================================================================
# Define working directory, and date-time
# =============================================================================
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24 hr [NO JXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24 hr [NO BXP]
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
RSITE = 'Flechtdorf'
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


# =============================================================================
# Set plotting parameters
# =============================================================================
PLOT_METHODS = False
PLOT_FIGS = True
SAVE_FIGS = False
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

# rprods = sorted(rprods_dp[1:] + rprods_hbr[1:] + rprods_opt + rprods_hyop
#                 + ['r_zo', 'r_kdpo', 'r_aho_kdpo'])

rprods = sorted(rprods_dp[1:] + rprods_hbr + rprods_opt + rprods_hyop
                + ['r_zo', 'r_kdpo'])

rvars = ['ZH [dBZ]', 'AH [dB/km]', 'alpha [-]', 'PhiDP [deg]', 'PIA [dB]',
         'KDP [deg/km]', 'ZH+ [dBZ]', 'ZDR [dB]', 'ADP [dB/km]',
         'beta [-]', 'KDP* [deg/km]', 'KDP+ [deg/km]']
vars2plot = ['ZH+ [dBZ]', 'KDP [deg/km]', 'r_kdpopt', 'r_zopt']

# %%
# =============================================================================
# Read-in QC PPI rdata
# =============================================================================
rqpe_dt = []
mlyrh = []
r_minsnr = []
rphidp0 = []
rzdr0 = []
rqpe_dt = []
gif_params = []
gif_georef = []
r_vars_all = {i: [] for i in rvars}
r_prods_all = {i+' Rainfall [mm/h]': [] for i in rprods}
# r_kdpopt_all = []
# radapt_coeffs = {'z_opt': [], 'kdp_opt': []}

# %%


rqpe_acch = [{rp: 0 for rp in rprods} for h1 in ds_fullgidx]
vld1 = [cnt for cnt, i1 in enumerate(RDQC_FILES) if type(i1) is list][0]

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
                # radapt_coeffs['kdp_opt'].append(
                #     (((rkdp_fit[0], rkdp_fit[1]))))
            if 'r_zopt' in rprods:
                rzh_fit = tpx.rzh_opt(
                    rpolara.vars[zh4rzo], rqpe.r_ah, rpolara.vars[ah4r],
                    # pia=rpolara.vars['PIA [dB]'], mlyr=rmlyr,
                    pia=None, mlyr=rmlyr,
                    # maxpia=(20 if rband == 'X' else 10),
                    maxpia=(50 if rband == 'X' else 50),
                    rzfit_b=(2.14 if rband == 'X' else 1.6),
                    rz_stv=[[next(item['rz_a'] for item in RPARAMS
                                  if item['site_name'] == rqpe.site_name),
                             next(item['rz_b'] for item in RPARAMS
                                  if item['site_name'] == rqpe.site_name)]],
                    plot_method=PLOT_METHODS)
                rqpe_opt.z_to_r(
                    rpolara.vars[zh4r], mlyr=rmlyr, a=rzh_fit[0],
                    b=rzh_fit[1],
                    beam_height=rpolara.georef['beam_height [km]'])
                rqpe.r_zopt = rqpe_opt.r_z
                # radapt_coeffs['z_opt'].append(((
                #                       (rqpe.r_zopt['coeff_a'],
                #                        rqpe.r_zopt['coeff_b']))))
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

            # for rp in rprods:
            #     if frad[0] == fhrad[0][0]:
            #         rqpe_acch[cnt][rp] = np.zeros(
            #             (resattc.params['nrays'],
            #              resattc.params['ngates']))
            #     if not np.isscalar(rqpe_acch[cnt][rp]):
            #         rqpe_acch[cnt][rp] = np.nansum(
            #             (rqpe_acch[cnt][rp],
            #              getattr(rqpe, rp)['Rainfall [mm/h]']), axis=0)
            rqpe_dt.append(rqpe.scandatetime)

            rad_georef = {}
            # rad_georef['theta'] = resattc.georef['theta']
            # rad_georef['rho'] = resattc.georef['rho']
            rad_georef['grid_rectx'] = rpolara.georef['grid_rectx']
            rad_georef['grid_recty'] = rpolara.georef['grid_recty']
            # rad_georef['grid_wgs84x'] = resattc.georef['grid_wgs84x']
            # rad_georef['grid_wgs84y'] = resattc.georef['grid_wgs84y']
            gif_params.append(rpolara.params)
            gif_georef.append(rad_georef)
            # r_zopt_all.append(rqpe.r_zopt['Rainfall [mm/h]'])
            # r_z_all.append(rqpe.r_z)
            # r_ah_all.append(rqpe.r_ah['Rainfall [mm/h]'])
            # r_adp_all.append(rqpe.r_adp['Rainfall [mm/h]'])
            # r_kdpopt_all.append(rqpe.r_kdpopt['Rainfall [mm/h]'])
            # r_z_ah_all.append(rqpe.r_z_ah['Rainfall [mm/h]'])
            # r_z_kdp_all.append(rqpe.r_z_kdp['Rainfall [mm/h]'])
            for nv in vars2plot:
                if nv in r_vars_all:
                    r_vars_all[nv].append(rpolara.vars[nv])
                elif nv + ' Rainfall [mm/h]' in r_prods_all:
                    r_prods_all[nv
                                + ' Rainfall [mm/h]'].append(
                                    getattr(rqpe, nv)['Rainfall [mm/h]'])
            rqpe_dt.append(rqpe.scandatetime)

# zh_all = []
# zdr_all = []
# kdp_all = []
# ah_all = []
# adp_all = []
# alpha_mean = []

# r_ah_all = []
# r_adp_all = []
# r_kdp_all = []
# r_z_ah_all = []
# r_z_kdp_all = []
# ims = []
# figs = []
# axs = []


# %%

rad_georef = gif_georef
rad_params = gif_params
# rpro1 = 'r_kdpopt_all'
# rpro1 = 'r_zopt_all'

# if rpro1 == 'r_zopt_all':
#     rad_vars = r_zopt_all
# elif rpro1 == 'r_kdpopt_all':
#     rad_vars = r_kdpopt_all
# var2plot='Rainfall [mm/h]'
var2plot = 'ZH+ [dBZ]'
# var2plot = 'KDP [deg/km]'
# var2plot = 'r_zopt Rainfall [mm/h]'
# var2plot = 'r_kdpopt Rainfall [mm/h]'
rpro1 = var2plot
rad_vars = r_vars_all | r_prods_all
rad_vars = rad_vars[var2plot]
# rpro1 = 'ZH+ [dBZ]'

fig_title = None
coord_sys = 'rect'
xlims = None
ylims = None
data_proj = None
ucmap = None
unorm = None
ring = None
range_rings = None
vars_bounds = None
cpy_feats = None
proj_suffix = 'osgb'
rd_maxrange = False
pixel_midp = False
points2plot = None
ptvar2plot = None
fig_title = None
fig_size = None
gifdir = '/home/dsanchez/sciebo_dsr/pd_rdres/anims/'
single_site = True
shpfile = None
data_info = False
cpy_feats = {'status': False}
data_proj = ccrs.PlateCarree()
proj_suffix = 'wgs84'
# fig_size=(13.3, 7)
RSITE = rqpe.site_name
# =============================================================================
lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
       'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-1, 3, 17],
       'rhoHV [-]': [0.3, .9, 1], 'V [m/s]': [-5, 5, 11],
       'gradV [dV/dh]': [-1, 0, 11], 'LDR [dB]': [-35, 0, 11],
       'Rainfall [mm/h]': [0.1, 64, 11], 'Rainfall [mm]': [0.1, 150, 14],
       'Beam_height [km]': [0, 6, 25]}
# =============================================================================
if vars_bounds is not None:
    lpv.update(vars_bounds)
bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
       if 'rhoHV' not in key
       else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                       np.linspace(value[1], value[2], 11)))
       for key, value in lpv.items()}
if vars_bounds is None:
    bnd['bRainfall [mm/h]'] = np.array((0.01, 0.5, 1, 2, 4, 8, 12, 20,
                                        28, 36, 48, 64, 80, 100))
# =============================================================================
    bnd['bRainfall [mm]'] = np.array((0.01, 1, 2, 4, 8, 12, 15, 20, 25,
                                      30, 40, 50, 75, 100, 150, 200))
# =============================================================================
dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                           mpl.colormaps['tpylsc_rad_pvars'].N,
                                           extend='both')
         for key, value in bnd.items()}
if 'bZH [dBZ]' in bnd.keys():
    dnorm['nZH [dBZ]'] = mcolors.BoundaryNorm(
        bnd['bZH [dBZ]'], mpl.colormaps['tpylsc_rad_ref'].N, extend='both')
if 'brhoHV [-]' in bnd.keys():
    dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(
        bnd['brhoHV [-]'], mpl.colormaps['tpylsc_rad_pvars'].N, extend='min')
if 'bRainfall [mm/h]' in bnd.keys():
    bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/h]'],
                                mpl.colormaps['tpylsc_rad_rainrt'].N,
                                extend='max')
    dnorm['nRainfall [mm/h]'] = bnrr
# =============================================================================
if 'bRainfall [mm]' in bnd.keys():
    bnrh = mcolors.BoundaryNorm(bnd['bRainfall [mm]'],
                                mpl.colormaps['tpylsc_rad_rainrt'].N,
                                extend='max')
    dnorm['nRainfall [mm]'] = bnrh
if 'bBeam_height [km]' in bnd.keys():
    bnbh = mcolors.BoundaryNorm(bnd['bBeam_height [km]'],
                                mpl.colormaps['gist_earth_r'].N,
                                extend='max')
    dnorm['nBeam_height [km]'] = bnbh
# =============================================================================
if 'bZDR [dB]' in bnd.keys():
    dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(
        bnd['bZDR [dB]'], mpl.colormaps['tpylsc_rad_2slope'].N, extend='both')
    # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
    #                                           vcenter=0,
    #                                           vmax=lpv['ZDR [dB]'][1],
if 'bKDP [deg/km]' in bnd.keys():
    dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(
        bnd['bKDP [deg/km]'], mpl.colormaps['tpylsc_rad_2slope'].N,
        extend='both')
if 'bV [m/s]' in bnd.keys():
    dnorm['nV [m/s]'] = mcolors.BoundaryNorm(
        bnd['bV [m/s]'], mpl.colormaps['tpylsc_div_dbu_rd'].N, extend='both')
# txtboxs = 'round, rounding_size=0.5, pad=0.5'
# txtboxc = (0, -.09)
# fc, ec = 'w', 'k'
if unorm is not None:
    dnorm.update(unorm)
# =============================================================================
cbtks_fmt = 2
# =============================================================================
if var2plot is None or var2plot == 'ZH [dBZ]':
    if 'ZH [dBZ]' in rad_vars.keys():
        cmaph, normp = mpl.colormaps['tpylsc_rad_ref'], dnorm['nZH [dBZ]']
        var2plot = 'ZH [dBZ]'
    else:
        var2plot = list(rad_vars.keys())[0]
        cmaph = mpl.colormaps['tpylsc_rad_pvars']
        normp = dnorm.get('n'+var2plot)
        if '[-]' in var2plot:
            cbtks_fmt = 2
            cmaph = mpl.colormaps['tpylsc_rad_pvars']
            tcks = bnd['brhoHV [-]']
        if '[dB]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rad_2slope']
            cbtks_fmt = 1
        if '[deg/km]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rad_2slope']
            # cbtks_fmt = 1
        if '[m/s]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_div_dbu_rd']
        if '[mm/h]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rad_rainrt']
            cbtks_fmt = 1
            cmaph.set_under('whitesmoke')
# =============================================================================
        if '[mm]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rad_rainrt']
            cmaph.set_under('whitesmoke')
            cbtks_fmt = 1
        if '[km]' in var2plot:
            cmaph = mpl.colormaps['gist_earth']
# =============================================================================
else:
    cmaph = mpl.colormaps['tpylsc_rad_pvars']
    normp = dnorm.get('n'+var2plot)
    if '[-]' in var2plot:
        cbtks_fmt = 2
        tcks = bnd['brhoHV [-]']
    if '[dB]' in var2plot:
        cmaph = mpl.colormaps['tpylsc_rad_2slope']
        cbtks_fmt = 1
    if '[dBZ]' in var2plot:
        cmaph = mpl.colormaps['tpylsc_rad_ref']
        normp = dnorm.get([k1 for k1 in dnorm if '[dBZ]' in k1][0])
        cbtks_fmt = 1
    if '[deg/km]' in var2plot:
        cmaph = mpl.colormaps['tpylsc_rad_2slope']
        # normp = [k1 for k1 in dnorm if '[deg/km]' in k1]
        normp = dnorm.get([k1 for k1 in dnorm if '[deg/km]' in k1][0])
    if '[m/s]' in var2plot:
        cmaph = mpl.colormaps['tpylsc_div_dbu_rd']
    if '[mm/h]' in var2plot:
        cmaph = mpl.colormaps['tpylsc_rad_rainrt']
        normp = dnorm.get([k1 for k1 in dnorm if '[mm/h]' in k1][0])
        cmaph.set_under('whitesmoke')
        # tpycm.set_under(color='#D2ECFA', alpha=0)
        # mpl.colormaps['tpylsc_rad_rainrt'].set_bad(color='#D2ECFA', alpha=0)
        cbtks_fmt = 1
# =============================================================================
    if '[mm]' in var2plot:
        cmaph = mpl.colormaps['tpylsc_rad_rainrt']
        cmaph.set_under('whitesmoke')
        cbtks_fmt = 1
    if '[km]' in var2plot:
        cmaph = mpl.colormaps['gist_earth']
# =============================================================================
if ucmap is not None:
    cmaph = ucmap

cpy_features = {'status': False,
                # 'coastresolution': '10m',
                'add_land': False,
                'add_ocean': False,
                'add_coastline': False,
                'add_borders': False,
                'add_countries': True,
                'add_provinces': True,
                'borders_ls': ':',
                'add_lakes': False,
                'lakes_transparency': 0.5,
                'add_rivers': False,
                'tiles': False,
                'tiles_source': None,
                'tiles_style': None,
                'tiles_res': 8, 'alpha_tiles': 0.5, 'alpha_rad': 1
                }
if cpy_feats:
    cpy_features.update(cpy_feats)
if cpy_features['status']:
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    countries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
        facecolor='none')

gfname = ('anim_polara_' + f'{RSITE}_'
          + rad_params[0]['datetime'].strftime("%Y%m%d%H%M%S_")
          + rad_params[-1]['datetime'].strftime("%Y%m%d%H%M%S")
          + rpro1)
# =============================================================================

# =============================================================================
if coord_sys == 'polar':
    if fig_size is None:
        fig_size = (6, 6.15)
    fig, ax1 = plt.subplots(figsize=fig_size,
                            subplot_kw=dict(projection='polar'))
elif coord_sys == 'rect' and cpy_features['status'] is False:
    # =====================================================================
    # ptitle = dtdes1 + dtdes2
    # =====================================================================
    if fig_size is None:
        fig_size = (6, 6.75)
    fig, ax1 = plt.subplots(figsize=fig_size)
elif coord_sys == 'rect' and cpy_features['status']:
    proj = ccrs.PlateCarree()
    if fig_size is None:
        fig_size = (9, 6)
    if data_proj:
        proj2 = data_proj
    # else:
    #     raise TowerpyError('User must specify the projected coordinate'
    #                        ' system of the radar data e.g.'
    #                        ' ccrs.OSGB(approx=False) or ccrs.UTM(zone=32)')
    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    ax1 = fig.add_subplot(projection=proj)

def animate(nanim):
    ax1.clear()
    # plt.cla()
    # dtdes0 = f"[{rad_params['site_name']}]"
    # dtdes1 = f"{rad_params[nanim]['elev_ang [deg]']:{2}.{3}} Deg."
    # dtdes2 = f"{rad_params[nanim]['datetime']:%Y-%m-%d %H:%M:%S}"
    if fig_title is None:
        if isinstance(rad_params[nanim]['elev_ang [deg]'], str):
            dtdes1 = f"{rad_params[nanim]['elev_ang [deg]']} -- "
        else:
            dtdes1 = f"{rad_params[nanim]['elev_ang [deg]']:{2}.{3}} deg. -- "
        dtdes2 = f"{rad_params[nanim]['datetime']:%Y-%m-%d %H:%M:%S}"
        ptitle = dtdes1 + dtdes2
    else:
        ptitle = fig_title
    if '[mm/h]' in var2plot:
        fig.suptitle(f'{ptitle} \n'
                     + f"PPI {var2plot[var2plot.find(' ')+1:]}"
                     + f" -- {RPRODSLTX.get(var2plot[:var2plot.find(' ')])}",
                     fontsize=14)
    else:
        fig.suptitle(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
    if coord_sys == 'polar':
        mappable = ax1.pcolormesh(rad_georef[nanim]['theta'],
                                  rad_georef[nanim]['rho'],
                                  np.flipud(rad_vars[nanim][var2plot]),
                                  shading='auto', cmap=cmaph, norm=normp)
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
        ax1.grid(color='gray', linestyle=':')
        ax1.set_theta_zero_location('N')
        ax1.tick_params(axis='both', labelsize=10)
        ax1.set_yticklabels([])
        ax1.set_thetagrids(np.arange(0, 360, 90))
        ax1.axes.set_aspect('equal')
        if var2plot == 'rhoHV [-]':
            cb1 = plt.colorbar(mappable, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp, ticks=tcks,
                               format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', axis='both', labelsize=10)
        else:
            cb1 = plt.colorbar(mappable, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp)
            cb1.ax.tick_params(direction='in', axis='both', labelsize=10)
        cb1.ax.set_title(f'{var2plot}', fontsize=10)
        plt.tight_layout()

    elif coord_sys == 'rect' and cpy_features['status'] is False:
        mappable = ax1.pcolormesh(rad_georef[nanim]['grid_rectx'],
                                  rad_georef[nanim]['grid_recty'],
                                  rad_vars[nanim],  # [var2plot],
                                  shading='auto',
                                  cmap=cmaph, norm=normp)
        if rd_maxrange:
            ax1.plot(rad_georef['grid_rectx'][:, -1],
                     rad_georef['grid_recty'][:, -1], 'gray')
        if pixel_midp:
            binx = rad_georef['grid_rectx'].ravel()
            biny = rad_georef['grid_recty'].ravel()
            ax1.scatter(binx, biny, c='grey', marker='+', alpha=0.2)
# =============================================================================
        if points2plot is not None:
            if len(points2plot) == 2:
                ax1.scatter(points2plot['grid_rectx'],
                            points2plot['grid_recty'], color='k',
                            marker='o', )
            elif len(points2plot) == 3:
                ax1.scatter(points2plot['grid_rectx'],
                            points2plot['grid_recty'], marker='o',
                            norm=normp, edgecolors='w',
                            c=[points2plot[ptvar2plot]], cmap=cmaph)
# =============================================================================
        if range_rings is not None:
            if isinstance(range_rings, (int, float)):
                nrings = np.arange(range_rings*1000,
                                   rad_georef['range [m]'][-1],
                                   range_rings*1000)
            elif isinstance(range_rings, (list, tuple)):
                nrings = np.array(range_rings) * 1000
            idx_rs = [find_nearest(rad_georef['range [m]'], r)
                      for r in nrings]
            dmmy_rsx = np.array([rad_georef['grid_rectx'][:, i]
                                 for i in idx_rs])
            dmmy_rsy = np.array([rad_georef['grid_recty'][:, i]
                                 for i in idx_rs])
            dmmy_rsz = np.array([np.ones(i.shape) for i in dmmy_rsx])
            ax1.scatter(dmmy_rsx, dmmy_rsy, dmmy_rsz, c='grey', ls='--',
                        alpha=3/4)
            ax1.axhline(0, c='grey', ls='--', alpha=3/4)
            ax1.axvline(0, c='grey', ls='--', alpha=3/4)
            ax1.grid(True)

        if ring is not None:
            idx_rr = find_nearest(rad_georef['range [m]'],
                                  ring*1000)
            dmmy_rx = rad_georef['grid_rectx'][:, idx_rr]
            dmmy_ry = rad_georef['grid_recty'][:, idx_rr]
            dmmy_rz = np.ones(dmmy_rx.shape)
            ax1.scatter(dmmy_rx, dmmy_ry, dmmy_rz, c='k', ls='--', alpha=3/4)
        if xlims is not None:
            ax1.set_xlim(xlims)
        if ylims is not None:
            ax1.set_ylim(ylims)
        ax1.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax1.set_ylabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax1.tick_params(direction='in', axis='both', labelsize=10)
        if nanim == 0:
            ax1_divider = make_axes_locatable(ax1)
            cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
            if var2plot == 'rhoHV [-]':
                cb1 = fig.colorbar(mappable, cax=cax1,
                                   orientation='horizontal', ticks=tcks,
                                   format=f'%.{cbtks_fmt}f')
                cb1.ax.tick_params(direction='in', labelsize=10)
            else:
                cb1 = fig.colorbar(mappable, cax=cax1,
                                   orientation='horizontal')
                cb1.ax.tick_params(direction='in', labelsize=12)
            cax1.xaxis.set_ticks_position('top')
        plt.tight_layout()


anim = animation.FuncAnimation(fig, animate, repeat=True,
                               repeat_delay=10, frames=len(rad_params),
                               interval=50, )
# plt.show()
if gifdir is not None:
    gfname = gfname[:gfname.find('[')-1]
    # gfname = gfname.replace('/', '')
    gfname = gfname.replace('+', 'p')
    anim.save(f'{gifdir}{gfname}.gif', fps=5, writer='imagemagick')


# from matplotlib.animation import FFMpegWriter
# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Movie support!')
# writer = FFMpegWriter(fps=15, metadata=metadata)

# fig = plt.figure()
# l, = plt.plot([], [], 'k-o')

# plt.xlim(-5, 5)
# plt.ylim(-5, 5)

# x0, y0 = 0, 0

# with writer.saving(fig, "writer_test.mp4", 100):
#     for i in range(100):
#         x0 += 0.1 * np.random.randn()
#         y0 += 0.1 * np.random.randn()
#         l.set_data(x0, y0)
#         writer.grab_frame()
