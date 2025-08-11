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
RADAR_SITE = 'Essen'
PTYPE = 'qvps'

fullqc = True
read_mlcal = True

PLOT_METHODS = False
SAVE_FIGS = False

DTWORK = dt.datetime(2017, 7, 24, 0, 0)  # 24hr [NO JXP]
DTWORK = dt.datetime(2017, 7, 25, 0, 0)  # 24hr [NO JXP]
# DTWORK = dt.datetime(2018, 5, 16, 0, 0)  # 24hr []
# DTWORK = dt.datetime(2018, 9, 23, 0, 0)  # 24 hr [NO JXP]
# DTWORK = dt.datetime(2018, 12, 2, 0, 0)  # 24 hr [NO JXP]
# DTWORK = dt.datetime(2019, 5, 8, 0, 0)  # 24 hr [NO JXP]
# DTWORK = dt.datetime(2019, 7, 20, 0, 0)  # 16 hr [NO BXP][JXP8am]
# DTWORK = dt.datetime(2020, 6, 17, 0, 0)  # 24 hr [NO BXP]
# DTWORK = dt.datetime(2021, 7, 13, 0, 0)  # 24 hr []
# DTWORK = dt.datetime(2021, 7, 14, 0, 0)  # 24 hr [NO BXP]

# DTWORK = dt.datetime(2014, 10, 7, 0, 0)  # 24hr []
# DTWORK = dt.datetime(2019, 5, 11, 0, 0)  # 24hr [NO JXP]
# DTWORK = dt.datetime(2019, 5, 20, 0, 0)  # 24hr [NO JXP] LOR
# DTWORK = dt.datetime(2020, 6, 13, 0, 0)  # 24 hr [NO BXP]
# DTWORK = dt.datetime(2020, 6, 14, 0, 0)  # 24 hr [NO BXP]
# DTWORK = dt.datetime(2021, 2, 6, 0, 0)  # 24 hr [NO BXP]
# DTWORK = dt.datetime(2023, 12, 23, 0, 0)  # 24 hr [NO BXP]

LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

extend_mlyr = False
if extend_mlyr:
    appx = '_extmlyr'
else:
    appx = ''
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

PPFILES = [WDIR+i for i in sorted(os.listdir(WDIR))
           if i.endswith(f'{PTYPE}.tpy') and RADAR_SITE in i
           and i.startswith(f"{DTWORK.strftime('%Y%m%d')}")]

RES_DIR = LWDIR + f"pd_rdres/qvps_d4calib{appx}/{DTWORK.strftime('%Y%m%d')}/"
if read_mlcal:
    RCFILES = [RES_DIR+i for i in sorted(os.listdir(RES_DIR))
               if i.endswith('qvps.tpy') and RADAR_SITE in i]

# =============================================================================
# %% Read radar profiles
# =============================================================================
with open(PPFILES[0], 'rb') as f:
    rprofs = pickle.load(f)

# rprofs = rprofs[288:]
if read_mlcal:
    with open(RCFILES[0], 'rb') as f:
        rprfc = pickle.load(f)
else:
    rprfc = []

# =============================================================================
# %% ZH Offset adjustment
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
# %% ZDR bias adjustment
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
# %% PhiDP bias adjustment
# =============================================================================
phidp_oc = False
if read_mlcal:
    phidpo = np.array([i.phidp_offset for i in rprfc['phidpO']])
if phidp_oc:
    for cnt, rp in enumerate(rprofs):
        rp.qvps['PhiDP [deg]'] -= phidpo[cnt]

# =============================================================================
# %% Adjust relative height
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

# %% Plot QVPs of PP
tz = 'Europe/Berlin'
htixlim = [DTWORK.replace(tzinfo=ZoneInfo(tz)),
           (DTWORK + dt.timedelta(seconds=86399)).replace(tzinfo=ZoneInfo(tz))]
htiylim = [0.29, 8.25]
# htiylim = [0., 12]

# tp.datavis.rad_display.plot_radprofiles(
#     rprofs[12], rprofs[12].georef['profiles_height [km]'], colours=False,
#     vars_bounds={'PhiDP [deg]': [-10, 20, 13]}, ylims=htiylim,
#     # stats='std_dev'
#     )

# %% Plot QVPs HTI
if fullqc:
    for rp1 in rprofs:
        rp1.qvps['ZH- [dBZ]'] = rp1.qvps['ZH+ [dBZ]'] - rp1.qvps['ZH [dBZ]']

# v2p = 'PhiDP [deg]'
v2p = 'AH [dB/km]'
# v2p = 'KDP [deg/km]'
v2p = 'ZH+ [dBZ]'
# v2p = 'ZHa [dBZ]'
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
if v2p == 'ZH+ [dBZ]' or v2p == 'ZHa [dBZ]':
    ucmap = 'tpylsc_rad_ref'
if v2p == 'ZH- [dBZ]':
    ucmap = 'tpylsc_div_rd_w_k'
# elif v2p == 'KDP [deg/km]' or v2p == 'AH [dB/km]':
    # contourl = 'ZH [dBZ]'
    # contourl = None
    # ptype = 'fcontour'
    # ptype = 'pseudo'
else:
    ptype = 'fcontour'
    ptype = 'pseudo'
    ucmap = None
    cbticks = None
    contourl = None

# ptype = 'fcontour'
# import seaborn as sns

cmap_prabhakar, unorm = tpx.unibonncmap()

kdplim = 0.5 if 'xpol' in RADAR_SITE else 0.2
radb = tp.datavis.rad_interactive.hti_base(
    rprofs, mlyrs=(rprfc['mlyr'] if read_mlcal else None),
    var2plot=v2p,
    stats=None,  # stats='std_dev',
    vars_bounds={'bin_class [0-5]': (0, 6, 7),
                 'prof_type [0-6]': (0, 7, 8),
                 'PhiDP [deg]': [0, 90, 10],
                 'KDP [deg/km]': [-kdplim, kdplim*3, 17],  # [-0.20, 0.6, 17],
                 # 'KDP [deg/km]': [-0.3, 0.6, 15],  # [-0.20, 0.6, 17],
                 'AH [dB/km]': [0., 0.050, 11],  # [0., 0.20, 21]
                 'ZDR [dB]': [-1.0, 3, 17],  # [0., 0.20, 21]
                 # 'ZH+ [dBZ]': [-10, 50, 13],
                 # 'ZHa [dBZ]': [-10, 60, 15],
                 # 'ZH- [dBZ]': [-8, 8, 15],
                 },
    cbticks=cbticks, contourl=contourl,
    ptype=ptype, htiylim=htiylim, htixlim=htixlim,
    # ucmap=ucmap,
    # ucmap=sns.color_palette("hls", 15, as_cmap=True),
    ucmap=cmap_prabhakar, unorm=unorm.get(v2p),
    fig_size=(19.2, 11), tz=tz,)
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


# %% CFADS

# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# import matplotlib as mpl

# # PLot CFAD or 2D histograms

# # =============================================================================
# # Method using matplotlib example
# # =============================================================================
# mpl.rcParams['xtick.labelsize'] = 10
# # mpl.rcParams.update(mpl.rcParamsDefault)

# var2plot = 'rhoHV [-]'
# # var2plot = 'ZDR [dB]'
# var2plot = 'ZH [dBZ]'
# # var2plot = 'PhiDP [deg]'
# var2plot = 'KDP [deg/km]'
# rvar = np.array([i.qvps[var2plot] for c, i in enumerate(rprofs) 
#                  if not np.isnan(rprfc['mlyr'][c].ml_top)])
# pheight = np.array([i.georef['profiles_height [km]'] for c, i in enumerate(rprofs) 
#                  if not np.isnan(rprfc['mlyr'][c].ml_top)])

# binsy = [0, 10, .5]
# binsx = [.6, 1, .01]
# binsx = [-.1, 0.6, .05]
# binsx = [-10, 60, .5]
# # binsx = [0, 180, 1]

# cmap = plt.colormaps['tpylsc_useq_morning_r']
# cmap = plt.colormaps['tpylsc_useq_fiery']
# # cmap = plt.colormaps['tpylsc_useq_bupkyw']
# # cmap = plt.colormaps['tpylsc_rad_pvars']
# cmap = plt.colormaps['Oranges']
# # cmap = cmap.with_extremes(bad=cmap(0))
# # bins_hist2d = 100
# # n1 = mpc.LogNorm(vmin=1, vmax=10)

# fig, axes = plt.subplots(ncols=4, figsize=(15, 10),  # layout='constrained',
#                          sharex=True, sharey=True)

# # =============================================================================
# # Plot series using `plot` and a small value of `alpha`
# # =============================================================================
# ax0 = axes[0]
# ax0.plot(rvar.T, pheight.T, color="C0", alpha=.5)
# # ax0.plot(rvar, pheight, color="C0", alpha=0.5)
# ax0.set_xlim(binsx[:-1])
# # ax0.set_ylim(binsy[:-1])
# ax0.set_ylim([0, binsy[1]])
# ax1_divider = make_axes_locatable(ax0)
# cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
# cax1.remove()
# ax0.grid()
# ax0.set_ylabel('Height [km]')
# ax0.set_xlabel(f'{var2plot}')

# # =============================================================================
# # Plot 2d histogram using contourf and linear color scale
# # =============================================================================
# # Linearly interpolate between the points in each time series
# # num_series = len(rvar)
# # num_fine = 5000
# # pheight_fine = np.linspace(pheight[:, 0].min(), pheight[:, 0].max(),
# #                            num_fine)
# # rvar_fine = np.concatenate([np.interp(pheight_fine, pheight[:, 0], y_row)
# #                             for y_row in rvar])
# # pheight_fine = np.broadcast_to(pheight_fine, (num_series, num_fine)).ravel()
# # rvar_fine = np.nan_to_num(rvar_fine, nan=-100)
# # hist, hbinsx, hbinsy = np.histogram2d(rvar_fine, pheight_fine, bins=300)
# m = ~np.isnan(rvar.ravel()) & ~np.isnan(pheight.ravel())
# hist, hbinsx, hbinsy = np.histogram2d(rvar.ravel()[m], pheight.ravel()[m],
#                                       # bins=[np.linspace(binsx[0], binsx[1],
#                                       #                   int(1+(binsx[1]-binsx[0])/binsx[2])),
#                                       #       np.linspace(binsy[0], binsy[1],
#                                       #                   int(1+(binsy[1]-binsy[0])/binsy[2]))]
#                                       # bins=bins_hist2d,
#                                       # range=[[-10, 60], [0, 10]]
#                                       bins=[np.arange(binsx[0], binsx[1],
#                                                       binsx[2]),
#                                             np.arange(binsy[0], binsy[1],
#                                                       binsy[2])],
#                                       )
# ax1 = axes[1]
# pcm = ax1.contourf(hbinsx[:-1], hbinsy[:-1],
#                    np.where(hist <= 0, 0, hist).T, cmap=cmap,
#                    # rasterized=True
#                    )
# ax1_divider = make_axes_locatable(ax1)
# cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
# cb1 = fig.colorbar(pcm, cax=cax1, extend='max', orientation="horizontal")
# cb1.ax.tick_params(which='both', direction='in', labelsize=11)
# cb1.ax.set_title('Counts', fontsize=11)
# cax1.xaxis.set_ticks_position("top")
# ax1.grid()
# ax1.set_xlabel(f'{var2plot}')

# # =============================================================================
# # Plot 2d histogram using pcolormesh and log colorscale
# # =============================================================================
# ax2 = axes[2]
# # You can tune vmax to make signal more visible
# bounds = np.linspace(0, 20, 21)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
# pcm = ax2.pcolormesh(hbinsx, hbinsy, hist.T, cmap=cmap,  # norm=norm,
#                      # norm=mpc.LogNorm(vmax=2.1e2),
#                      rasterized=True)
# ax2_divider = make_axes_locatable(ax2)
# cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
# cb2 = fig.colorbar(pcm, cax=cax2, extend='max', orientation="horizontal")
# cb2.ax.tick_params(which='both', direction='in', labelsize=11)
# cb2.ax.set_title('Counts', fontsize=11)
# cax2.xaxis.set_ticks_position("top")
# # ax2.set_xlabel('$Z_H$ [dBZ]')
# ax2.set_xlabel(f'{var2plot}')
# ax2.grid()
# # ax2.set_xlim(hbinsx[:-1])
# # ax2.set_ylim(hbinsy[:-1])

# # =============================================================================
# # Plot relative 2d histogram using contourf
# # =============================================================================
# ax = axes[3]
# hbinsxr = (hbinsx[0:-1] + hbinsx[1:len(hbinsx)])/2
# hbinsyr = (hbinsy[0:-1] + hbinsy[1:len(hbinsy)])/2
# nsum = hist.max()
# # nsum = len(pheight)
# bounds = np.linspace(0, 1000, 11)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')
# histn = 100*(hist/nsum)
# # pcm = ax.contourf(hbinsx[:-1], hbinsy[:-1],
# #                   np.where(histn <= 0, np.nan, histn).T, 21, cmap=cmap,
# #                   shading='auto', rasterized=True  # norm=norm,
# #                   )
# pcm = ax.hist2d(rvar.ravel()[m], pheight.ravel()[m],
#                   # bins=[np.arange(-10, 60, .1),
#                   #       np.arange(0, 10, .5)],
#                   bins=[
#                       # np.arange(-0.1, .5, 0.01),
#                         np.arange(binsx[0], binsx[1], binsx[2]),
#                         # np.arange(-0.5, 2.5, .01),
#                         # np.arange(0.6, 1.1, .01),
#                         np.arange(binsy[0], binsy[1], binsy[2])],
#                   cmap=cmap,
#                   shading='auto',
#                   norm=norm,
#                   )
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("top", size="7%", pad="2%")
# cb = fig.colorbar(pcm[3], cax=cax, extend='max', orientation="horizontal")
# cb.ax.tick_params(which='both', direction='in', labelsize=11)
# cb.ax.set_title('Frequency [%]', fontsize=11)
# cax.xaxis.set_ticks_position("top")

# ax.set_xlabel(f'{var2plot}')
# ax.grid()


# # pcm = axes[1].hist2d(np.ma.masked_invalid(x_fine.ravel()),
# #                       np.ma.masked_invalid(y_fine.ravel()),
# #                       range=[[0.01, 15], [-10, 60]], bins=bins_hist2d,
# #                       cmap=cmap, norm=n1)
# # fig.colorbar(pcm[3], ax=axes[1], label="# points", pad=0)
# # pcm = axes[1].contourf(xedges[:-1], yedges[:-1], h.T, cmap=cmap,# rasterized=True,
# #                        norm=mcolors.LogNorm(vmax=1.1e1),
# #                        )
# # pcm = axes[1].hexbin(xedges, yedges, gridsize=20, cmap='rainbow')