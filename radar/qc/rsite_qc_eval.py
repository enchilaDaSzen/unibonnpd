#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:18:21 2025

@author: dsanchez
"""

import datetime as dt
# from zoneinfo import ZoneInfo
import os
import pickle
import numpy as np
import towerpy as tp
from towerpy.utils import unit_conversion as tpuc
from towerpy.utils.radutilities import linspace_step
from radar import twpext as tpx
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from radar.rparams_dwdxpol import RPARAMS
from tqdm import tqdm

# =============================================================================
# Define working directory, and date-time
# =============================================================================
START_TIME = dt.datetime(2017, 7, 24, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2017, 7, 25, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2018, 5, 16, 0, 0)  # 24h []
# START_TIME = dt.datetime(2018, 9, 23, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2019, 5, 8, 0, 0)  # 24h [NO JXP]
# # START_TIME = dt.datetime(2019, 5, 11, 0, 0)  # 24h [NO JXP]
# START_TIME = dt.datetime(2019, 7, 20, 8, 0)  # 16h [NO BXP]
# START_TIME = dt.datetime(2020, 6, 17, 0, 0)  # 24h [NO BXP]
# START_TIME = dt.datetime(2021, 7, 13, 0, 0)  # 24h [NO BXP]
START_TIME = dt.datetime(2021, 7, 14, 0, 0)  # 24h [NO BXP]

# STOP_TIME = dt.datetime(2019, 7, 21, 0, 0)
# EVNTD_HRS = round((STOP_TIME - START_TIME).total_seconds() / 3600)
EVNTD_HRS = (16 if START_TIME == dt.datetime(2019, 7, 20, 8, 0) else 24)
STOP_TIME = START_TIME + dt.timedelta(hours=EVNTD_HRS)

QPE_TRES = dt.timedelta(minutes=5)

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
RSITE = 'Juxpol'
RPARAMS = [next(item for item in RPARAMS if item['site_name'] == RSITE)]

# =============================================================================
# Read-in QVPs data
# =============================================================================
data4calib = 'qvps'
DIRPROFSCAL = (LWDIR + 'pd_rdres/qvps_d4calib/'
               + f"{START_TIME.strftime('%Y%m%d')}/")
RCAL_FILES = {rs['site_name']:
              [DIRPROFSCAL+n for n in sorted(os.listdir(DIRPROFSCAL))
              if data4calib in n and rs['site_name'] in n] for rs in RPARAMS}

profs_data = {}
for k1, rs in RCAL_FILES.items():
    with open(rs[0], 'rb') as breader:
        profs_data[k1] = pickle.load(breader)

mlt_avg = np.nanmean([i.ml_top for i
                      in profs_data[RPARAMS[0]['site_name']]['mlyr']])
mlk_avg = np.nanmean([i.ml_thickness for i
                      in profs_data[RPARAMS[0]['site_name']]['mlyr']])

# =============================================================================
# Set plotting parameters
# =============================================================================
SAVE_FIGS = True
fig_size = (13, 7)
fig_size = (10.5, 7)
xlims, ylims = [4.15, 11.], [48.55, 52.75]  # DWDXPOL RADCOV
# xlims, ylims = [4.3, 11.], [48.5, 52.8]  # DWDXPOL COV
# xlims, ylims = [5.9, 10.31], [49.3, 52.3]  # PAPER
# xlims, ylims = [5.85, 11.], [48.55, 52.75]  # DWDXPOL DE

RES_DIR = LWDIR + f"pd_rdres/qpe_{START_TIME.strftime('%Y%m%d')}/rsite_qpe/"

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

# %%
# =============================================================================
# Set QPE parameters
# =============================================================================
temp = 15
z_thld = 40
filter_aml = True
if filter_aml:
    appxf = ''
else:
    appxf = '_amlb'

# Computing several theoretical rain estimation

rband = RPARAMS[0]['rband']

if rband == 'C':
    ahzh_a, ahzh_b = 2.49e-05, 0.755  # MD 15C
    kdpzh_zdr1_a, kdpzh_zdr1_b = 0, 0  # Guang Wen et al (2019)
    kdpzh_zdr2_a0, kdpzh_zdr2_a1 = 6.746, -2.97  # Gourley et al (2009)
    kdpzh_zdr2_a2, kdpzh_zdr2_a3 = 0.711, -0.079  # Gourley et al (2009)
    kdpzh_zdr3_a, kdpzh_zdr3_b = 1.82e-4, -1.28  # Gorgucci et al. (1992)
    kdpmean_a = (1.05*10**-4+1.13*10**-4+1.62*10**-4)/3
    kdpmean_b = (.908+.892+.85)/3
elif rband == 'X':
    ahzh_a, ahzh_b = 9.745e-05, 0.8  # MD15C
    kdpzh_zdr1_a, kdpzh_zdr1_b = 1.2e-4, -4.1e-5  # Guang Wen et al (2019)
    kdpzh_zdr2_a0, kdpzh_zdr2_a1 = 11.74, -4.02  # Gourley et al (2009)
    kdpzh_zdr2_a2, kdpzh_zdr2_a3 = -0.14, 0.13  # Gourley et al (2009)
    kdpzh_zdr3_a, kdpzh_zdr3_b = 0, 0  # Gorgucci et al. (1992)
    kdpmean_a = 1.01*10**-3
    kdpmean_b = 0.67

ahkdp_a = next(item['ahkdp_a'] for item in RPARAMS
               if item['site_name'] == RPARAMS[0]['site_name'])
ahkdp_b = next(item['ahkdp_b'] for item in RPARAMS
               if item['site_name'] == RPARAMS[0]['site_name'])
adpkdp_b = next(item['adpkdp_b'] for item in RPARAMS
                if item['site_name'] == RPARAMS[0]['site_name'])
adpkdp_a = next(item['adpkdp_a'] for item in RPARAMS
                if item['site_name'] == RPARAMS[0]['site_name'])
zdrzh_a = next(item['zdrzh_a'] for item in RPARAMS
               if item['site_name'] == RPARAMS[0]['site_name'])
zdrzh_b = next(item['zdrzh_b'] for item in RPARAMS
               if item['site_name'] == RPARAMS[0]['site_name'])

kdpii = linspace_step(0.01, 10, 0.01)
ahii = linspace_step(0.01, 1, 0.001)
adpii = linspace_step(0.01, 1, 0.001)
ahzhii = linspace_step(-10, 60, 0.01)
zdrzhii = linspace_step(-10, 60, 0.01)
zhkdpii = linspace_step(-10, 60, 0.01)
kdpzh_zdrii = linspace_step(-1, 4, 0.01)
kdpzh_zdrii2 = linspace_step(-1, 1.6, 0.01)


ahi = ahkdp_a * kdpii ** ahkdp_b
adpi = adpkdp_a * kdpii ** adpkdp_b
ahzhlii = tpuc.xdb2x(ahzhii)
ahzhi = ahzh_a * ahzhlii ** ahzh_b
zdrzhi = zdrzh_a * zdrzhii ** zdrzh_b
# (KDP/Zh)-ZDR linear Guang Wen et al 2019 XBAND only
kdpzh_zdril = kdpzh_zdr1_a + kdpzh_zdr1_b*kdpzh_zdrii2
# (KDP/Zh)-ZDR 3dP Gourley et al 2009
kdpzh_zdri3dp = (10**-5)*(kdpzh_zdr2_a0 + kdpzh_zdr2_a1*kdpzh_zdrii
                          + kdpzh_zdr2_a2*kdpzh_zdrii**2
                          + kdpzh_zdr2_a3*kdpzh_zdrii**3)
# (KDP/Zh)-ZDR 3dP Gorgucci et al. (1992) C-band only
kdpzh_zdrisc = kdpzh_zdr3_a*kdpzh_zdrii**kdpzh_zdr3_b
# KDP mean (Rhyzkov)
kdpmean_ii = kdpmean_a * tpuc.xdb2x(zhkdpii) ** kdpmean_b

# %%
adp_all = []
ah_all = []
kdpa_all = []
kdpp_all = []
zha_all = []
zhp_all = []
zdr_all = []
# alpha_mean = []

# iraf = RDQC_FILES[210]
for cnt, iraf in enumerate(tqdm(RDQC_FILES, desc='Gathering radar vars')):
    if iraf is not np.nan:
        with open(iraf, 'rb') as f:
            resattc = pickle.load(f)
            rmlyr = tp.ml.mlyr.MeltingLayer(resattc)
            rmlyr.ml_top = resattc.ml_top
            rmlyr.ml_bottom = resattc.ml_bottom
            rmlyr.ml_thickness = resattc.ml_thickness
            # PPI MLYR
            rmlyr.ml_ppidelimitation(resattc.georef, resattc.params,
                                     resattc.vars)
            # r_minsnr.append(resattc.min_snr)
            # rphidp0.append(resattc.phidp0)
            # rzdr0.append(resattc.zdr_offset)
            rband = next(item['rband'] for item in RPARAMS
                         if item['site_name'] == resattc.site_name)
            # TODO: FOR C BAND ZH(ATTC) WORKS BETTER FOR KDP, WHY?
            # if rband == 'C':
            #     zhr_kdp = 'ZH [dBZ]'  # ZH(ATTC)
            # else:
            #     zhr_kdp = 'ZH+ [dBZ]'  # ZH(AH)
            zha = 'ZH [dBZ]'  # ZH(ATTC)
            zhp = 'ZH+ [dBZ]'  # ZH(AH)
            zdra = 'ZDR [dB]'
            aha = 'AH [dB/km]'
            adpa = 'ADP [dB/km]'
            kdpa = 'KDP [deg/km]'  # AH
            kdpp = 'KDP+ [deg/km]'  # Vulpiani+AH
            if filter_aml:
                adp_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[adpa], np.nan)
                ah_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[aha], np.nan)
                kdpa_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[kdpa], np.nan)
                kdpp_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[kdpp], np.nan)
                zha_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[zha], np.nan)
                zhp_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[zhp], np.nan)
                zdr_data = np.where(
                    (rmlyr.mlyr_limits['pcp_region [HC]'] == 1),
                    resattc.vars[zdra], np.nan)
            else:
                adp_data = resattc.vars[adpa]
                ah_data = resattc.vars[aha]
                kdpa_data = resattc.vars[kdpa]
                kdpp_data = resattc.vars[kdpp]
                zha_data = resattc.vars[zha]
                zdr_data = resattc.vars[zdra]
            adp_all.append(adp_data)
            ah_all.append(ah_data)
            kdpa_all.append(kdpa_data)
            kdpp_all.append(kdpp_data)
            zha_all.append(zha_data)
            zhp_all.append(zhp_data)
            zdr_all.append(zdr_data)

adp_all = np.vstack(adp_all)
ah_all = np.vstack(ah_all)
kdpa_all = np.vstack(kdpa_all)
kdpp_all = np.vstack(kdpp_all)
zha_all = np.vstack(zha_all)
zhp_all = np.vstack(zhp_all)
zdr_all = np.vstack(zdr_all)

# %%
# =============================================================================
# Plot
# =============================================================================
lblsize = 18
tcksize = 14
cmap = 'Spectral_r'

zh_fc = True

if zh_fc:
    zh4plots = zhp_all
else:
    zh4plots = zha_all

n1 = mpc.LogNorm(vmin=1, vmax=10000)
# n1 = mpc.PowerNorm(gamma=0.5)
# bounds = np.linspace(0, 10000, 21)
# bounds = np.geomspace(1, 100000, num=11)
# n1 = mpc.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
gridsize = 500
zhlims = [10, 60]
zdrlims = (-1, 6)
if rband == 'C':
    # kdplims = [0, 1]  # CBAND LR
    # ahlims = [0, 0.1]  # CBAND LR
    # adplims = [0, 0.05]  # CBAND LR
    # kdplims = [0, 2]  # CBAND MR
    # ahlims = [0, 0.25]  # CBAND MR
    kdplims = [0, 6]  # CBAND HR
    ahlims = [0, 1]  # CBAND HR
    adplims = [0, 0.25]  # CBAND HR
    kdpzhlims = [0, 1e-3]
elif rband == 'X':
    kdplims = [0, 6]
    ahlims = [0, 2]  # XBAND MR
    ahlims = [0, 4]  # XBAND HR
    adplims = [0, 0.5]  # XBANDboxpol
    kdpzhlims = [0, 1e-3]

plt.style.use('default')
# plt.style.use('seaborn-v0_8')
lbl_size = 14
lbl_size2 = 12
fig, axs = plt.subplots(3, 2, figsize=(19.2, 11.4), sharex=False)
fig.suptitle(f"{RSITE} -- {START_TIME.strftime('%Y-%m-%d')} {EVNTD_HRS}H",
             fontsize=16)

# =============================================================================
# ZH_AH
# =============================================================================
# zhah_med = np.array([np.nanmedian(np.ma.masked_invalid(ah_all.ravel())[(np.ma.masked_invalid(zh4plots.ravel()) >= i) & (np.ma.masked_invalid(zh4plots.ravel()) <= i+0.1)])
#                     for i in ahzhii])
ax = axs[0, 0]
ax.plot(ahzhii, ahzhi, c='k', ls='--',
        # label=f'$A_H={ahzh_a:,.2e} Z_H^{{{ahzh_b:,.2f}}}$'
        # + ' [Diederich et al. (2015)]')
        label=(rf'$A_H={ahzh_a:,.2e}').replace('e', '\\times 10^{')+'}$'
        + f'$Z_H^{{{ahzh_b:,.2f}}}$ [Diederich et al. (2015)]')
hxb = ax.hexbin(np.ma.masked_invalid(zh4plots.ravel()),
                np.ma.masked_invalid(ah_all.ravel()), gridsize=gridsize,
                mincnt=1, cmap=cmap, norm=n1)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("top", size="7%", pad="2%")
cb = fig.colorbar(hxb, cax=cax, extend='max', orientation="horizontal")
cb.ax.tick_params(which='both', direction='in', labelsize=lbl_size)
cb.ax.set_title('Counts', fontsize=lbl_size)
cax.xaxis.set_ticks_position("top")
# cax.remove()
ax.set_xlim(zhlims)
ax.set_ylim(ahlims)
ax.tick_params(axis='both', which='major', labelsize=lbl_size2)
ax.set_xlabel('$Z_H$ [dBZ]', fontsize=lbl_size)
ax.set_ylabel('$A_H$ [dB/km]', fontsize=lbl_size)
ax.legend(loc='upper left', fontsize=lbl_size)
ax.grid()
ax.set_axisbelow(True)

# =============================================================================
# ZH_KDP
# =============================================================================
ax = axs[0, 1]
# ax = axs[1]
hxb = ax.hexbin(np.ma.masked_invalid(zh4plots.ravel()),
                np.ma.masked_invalid(kdpa_all.ravel()),
                gridsize=gridsize, mincnt=1, cmap=cmap, norm=n1)
if rband == 'X':
    ax.plot(zhkdpii, kdpmean_ii, c='k', ls='--',
            label=f'$K^{{mean}}_{{DP}}={kdpmean_a:,.2e}'
            + f'Z_{{H}}^{{{kdpmean_b:,.2f}}}$ [Ryzhkov A.]')
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("top", size="7%", pad="2%")
cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# cb.ax.tick_params(direction='in', labelsize=tcksize)
# cb.ax.set_title('Counts', fontsize=tcksize)
cb.ax.tick_params(which='both', direction='in', labelsize=lbl_size)
cb.ax.set_title('Counts', fontsize=lbl_size)
cax.xaxis.set_ticks_position("top")
# cax.remove()
ax.set_xlim(zhlims)
ax.set_ylim(kdplims)
ax.tick_params(axis='both', which='major', labelsize=lbl_size2)
ax.set_xlabel('$Z_{H}$ [dBZ]', fontsize=lbl_size)
ax.set_ylabel('$K_{DP}$ [deg/km]', fontsize=lbl_size)
ax.legend(fontsize=lbl_size)
ax.grid()
ax.set_axisbelow(True)

# =============================================================================
# ZH_ZDR
# =============================================================================
ax = axs[1, 0]
# ax = axs[0]
ax.plot(zdrzhii, zdrzhi, c='k', ls='--',
        # label=f'$Z_{{DR}}={zdrzh_a:,.2e}Z_{{H}}^{{{zdrzh_b:,.2f}}}$'
        # + '[Chen et al. (2021)]')
        label=(rf'$Z_{{DR}}={zdrzh_a:,.2e}').replace('e', '\\times 10^{')+'}$'
        + f'$Z_{{H}}^{{{zdrzh_b:,.2f}}}$' + '[Chen et al. (2021)]')
hxb = ax.hexbin(np.ma.masked_invalid(zh4plots.ravel()),
                np.ma.masked_invalid(zdr_all.ravel()),
                gridsize=gridsize, mincnt=1, cmap=cmap, norm=n1)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("top", size="7%", pad="2%")
# cb = fig.colorbar(hxb, cax=cax, extend='max', orientation="horizontal")
# cb.ax.tick_params(which='both', direction='in', labelsize=tcksize)
# cb.ax.set_title('Counts', fontsize=tcksize)
# cax.xaxis.set_ticks_position("top")
cax.remove()
ax.set_xlim(zhlims)
ax.set_ylim(zdrlims)
ax.tick_params(axis='both', which='major', labelsize=lbl_size2)
ax.set_xlabel('$Z_H$ [dBZ]', fontsize=lbl_size)
ax.set_ylabel('$Z_{DR}$ [dB]', fontsize=lbl_size)
ax.legend(loc='upper left', fontsize=lbl_size)
ax.grid()
ax.set_axisbelow(True)

# =============================================================================
# ZDR_KDP/ZH
# =============================================================================
ax = axs[1, 1]
kdpzh_zdri3dp2 = kdpzh_zdri3dp
# kdpzh_zdri3dp2 = (10**5)*kdpzh_zdri3dp
# kdpzh_zdri3dp2 = np.log(kdpzh_zdri3dp)
# kdpzh_zdri3dp2 = 10*np.log10(kdpzh_zdri3dp)
if rband == 'X':
    kdpzh_zdril2 = kdpzh_zdril
    # kdpzh_zdril2 = (10**5)*kdpzh_zdril
    # kdpzh_zdril2 = np.log(kdpzh_zdril)
    # kdpzh_zdril2 = 10*np.log10(kdpzh_zdril)
    ax.plot(kdpzh_zdrii, kdpzh_zdri3dp2, ls='--', c='k',
            label='$(K_{DP}/Z_h)=10^{-5}$'
            + f'({kdpzh_zdr2_a0:+,.2f}{kdpzh_zdr2_a1:+,.2f} $Z_{{DR}}$'
            + f'{kdpzh_zdr2_a2:+,.2f} $Z_{{DR}}^2$'
            + f'{kdpzh_zdr2_a3:+,.2f} $Z_{{DR}}^3)$ [Gourley et al. (2009)]')
    ax.plot(kdpzh_zdrii2, kdpzh_zdril2, ls=':', c='k',
            label=(rf'$(K_{{DP}}/Z_h)={kdpzh_zdr1_a:,.1e}').replace('e', '\\times 10^{')+'}$'
            + (rf'${kdpzh_zdr1_b:,.1e}').replace('e', '\\times 10^{')+'}$'
            + '$Z_{{DR}}$ [Matrosov (2010)]')
if rband == 'C':
    kdpzh_zdrisc2 = kdpzh_zdrisc
    # kdpzh_zdrisc2 = (10**5)*kdpzh_zdrisc
    # kdpzh_zdrisc2 = np.log(kdpzh_zdrisc)
    # kdpzh_zdrisc2 = 10*np.log10(kdpzh_zdrisc)
    ax.plot(kdpzh_zdrii, kdpzh_zdrisc2, ls='-', c='k',
            label=f'$(K_{{DP}}/Z_h^{{0.95}})$={kdpzh_zdr3_a:,.2e}'
            + f'$Z_{{DR}}^{{{kdpzh_zdr3_b}}}$ [Gorgucci et al. (1992)]')
    ax.plot(kdpzh_zdrii, kdpzh_zdri3dp2, ls='--', c='k',
            label='$(K_{DP}/Z_h)=10^{-5}$'
            + f'({kdpzh_zdr2_a0:+,.2f}{kdpzh_zdr2_a1:+,.2f} $Z_{{DR}}$'
            + f'{kdpzh_zdr2_a2:+,.2f} $Z_{{DR}}^2$'
            + f'{kdpzh_zdr2_a3:+,.2f} $Z_{{DR}}^3)$ [Gourley et al. (2009)]')
hxb = ax.hexbin(np.ma.masked_invalid(zdr_all.ravel()),
                # np.ma.masked_invalid(10*np.log10(kdpa_all.ravel()) - zh_all.ravel()),
                # np.ma.masked_invalid(10*np.log10(kdpa_all.ravel() / tpuc.xdb2x(zh_all.ravel()))),
                # np.ma.masked_invalid(10*np.log10(kdpa_all.ravel() / tpuc.xdb2x(zh_all.ravel())**0.95)),
                # np.ma.masked_invalid(np.log(kdpa_all.ravel() / tpuc.xdb2x(zh_all.ravel()))),
                # np.ma.masked_invalid((kdpa_all.ravel() / tpuc.xdb2x(zh_all.ravel()))),
                # np.ma.masked_invalid((kdpa_all.ravel() / tpuc.xdb2x(zh_all.ravel()))*(10**5)),
                # np.ma.masked_invalid((kdpa_all.ravel() / tpuc.xdb2x(zh_all.ravel()))/(10**-5)),
                np.ma.masked_invalid((kdpa_all.ravel() / tpuc.xdb2x(zh4plots.ravel()))),
                # np.ma.masked_invalid((kdpa_all.ravel() / tpuc.xdb2x(zhp_all.ravel())**0.95)),
                # np.ma.masked_invalid(zdr_all.ravel()),
                gridsize=gridsize*2, mincnt=1, cmap=cmap, norm=n1)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("top", size="7%", pad="2%")
# cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# cb.ax.tick_params(direction='in', labelsize=tcksize)
# cax.xaxis.set_ticks_position("top")boxpol
cax.remove()
ax.set_xlim(zdrlims)
ax.set_ylim(kdpzhlims)
# ax.set_xlim(0, 0.001)
# ax.set_ylim([0, 10])
ax.set_xlabel('$Z_{DR}$ [dB]', fontsize=lbl_size)
ax.set_ylabel('$K_{DP}/Z_{h}$ [dB/km]', fontsize=lbl_size)
# ax.set_xlabel('$10log(K_{DP}/Z_{h})$')
# ax.set_ylabel('$log(K_{DP}/Z_{h})$')
# ax3.set_xlim(0, 6)
ax.tick_params(axis='both', which='major', labelsize=lbl_size2)
ax.legend(loc='upper right', fontsize=lbl_size)
ax.grid()
ax.set_axisbelow(True)

# =============================================================================
# KDP_AH
# =============================================================================
ax = axs[2, 0]
ax.plot(kdpii, ahi, label=f'$A_H={ahkdp_a:,.2f}K_{{DP}}^{{{ahkdp_b}}}$', c='k',
        ls='--')
hxb = ax.hexbin(np.ma.masked_invalid(kdpa_all.ravel()),
                np.ma.masked_invalid(ah_all.ravel()),
                gridsize=gridsize, mincnt=1, cmap=cmap, norm=n1)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("top", size="7%", pad="2%")
# cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# cb.ax.tick_params(direction='in', labelsize=tcksize)
# cb.ax.set_title('Counts', fontsize=tcksize)
# cax.xaxis.set_ticks_position("top")
cax.remove()
ax.set_xlabel('$K_{DP}$ [deg/km]', fontsize=lbl_size)
ax.set_ylabel('$A_H$ [dB/km]', fontsize=lbl_size)
ax.set_xlim(kdplims)
ax.set_ylim(ahlims)
ax.tick_params(axis='both', which='major', labelsize=lbl_size2)
# ax.set_ylim([0, 0.75])
ax.legend(loc='upper left', fontsize=lbl_size)
ax.grid()
ax.set_axisbelow(True)

# =============================================================================
# KDP_ADP
# =============================================================================
ax = axs[2, 1]
ax.plot(kdpii, adpi, c='k', ls='--',
        label=f'$A_{{DP}}={adpkdp_a:,.2f}K_{{DP}}^{{{adpkdp_b}}}$')
hxb = ax.hexbin(np.ma.masked_invalid(kdpa_all.ravel()),
                np.ma.masked_invalid(adp_all.ravel()),
                gridsize=gridsize, mincnt=1, cmap=cmap, norm=n1)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("top", size="7%", pad="2%")
# cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# cb.ax.tick_params(direction='in', labelsize=tcksize)
# cb.ax.set_title('Counts', fontsize=tcksize)
# cax.xaxis.set_ticks_position("top")
cax.remove()
ax.set_xlabel('$K_{DP}$ [deg/km]', fontsize=lbl_size)
ax.set_ylabel('$A_{DP}$ [dB/km]', fontsize=lbl_size)
ax.set_xlim(kdplims)
ax.set_ylim(adplims)
ax.tick_params(axis='both', which='major', labelsize=lbl_size2)
ax.legend(loc='upper left', fontsize=lbl_size)
ax.grid()
ax.set_axisbelow(True)

# =============================================================================
# ZDR_AH
# =============================================================================
# ax = axs[1, 1]
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("top", size="7%", pad="2%")
# hxb = ax.hexbin(np.ma.masked_invalid(zdr_all.ravel()),
#                 # np.ma.masked_invalid(adp_all.ravel()/kdp_all.ravel()),
#                 np.ma.masked_invalid(ah_all.ravel()), gridsize=gridsize,
#                 mincnt=1, cmap=cmap, norm=n1)
# # cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# # cb.ax.tick_params(direction='in', labelsize=tcksize)
# # cax.xaxis.set_ticks_position("top")
# ax.set_xlim(zdrlims)
# ax.set_ylim(ahlims)
# ax.set_xlabel('$Z_{DR}$ [dB]')
# ax.set_ylabel('$A_H$ [dB/deg]')
# ax.grid()

# =============================================================================
# ZDR_AH/KDP
# =============================================================================
# plot_type = 'lin'
# ax = axs[1, 1]
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("top", size="7%", pad="2%")
# hxb = ax.hexbin(
#     np.ma.masked_invalid(zdr_all.ravel()),
#     (np.ma.masked_invalid(adp_all.ravel())
#      / np.ma.masked_invalid(kdp_all.ravel())),
#     gridsize=gridsize*2, mincnt=1, cmap=cmap,np.log( norm=n1)
# # cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# ax.set_xlim(zdrlims)
# # ax.set_ylim([0, .15])
# # cb.ax.tick_params(direction='in', labelsize=tcksize)
# # cax.xaxis.set_ticks_position("top")
# cax.remove()
# ax.set_xlabel('$Z_{DR}$ [dB]')
# ax.set_ylabel('$A_{H} / K_{DP}$ [dB/deg]')
# # ax.legend()
# ax.grid()

# =============================================================================
# ZDR_ADP
# =============================================================================
# ax = axs[1, 1]
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("top", size="7%", pad="2%")
# hxb = ax.hexbin(np.ma.masked_invalid(zdr_all.ravel()),
#                 np.ma.masked_invalid(adp_all.ravel()), gridsize=gridsize,
#                 mincnt=1, cmap=cmap, norm=n1)
# # cb = fig.colorbar(hxb, cax=cax, extend='max', orientation='horizontal')
# ax.set_xlim(zdrlims)
# ax.set_ylim([0, 1])
# # ax.set_ylim([0, 0.5])
# # cb.ax.tick_params(direction='in', labelsize=tcksize)
# # cax.xaxis.set_ticks_position("top")
# cax.remove()
# ax.set_xlabel('$Z_{DR}$ [dB]')
# ax.set_ylabel('$A_{DP}$ [dB/km]')
# # ax.legend()
# ax.grid()

plt.tight_layout()

if SAVE_FIGS:
    if zh_fc:
        sfx = ''
    else:
        sfx = 'nfc'
    RES_DIR2 = RES_DIR.replace('rsite_qpe', 'rsite_dailyqc')
    fname = (f"{START_TIME.strftime('%Y%m%d')}"
             + f"_{RPARAMS[0]['site_name'][:3].lower()}_daily_rvars{sfx}.png")
    plt.savefig(RES_DIR2 + fname, format='png')
