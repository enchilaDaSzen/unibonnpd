#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:51:02 2025

@author: dsanchez
"""

import os
import datetime as dt
import pickle
import numpy as np
from radar import twpext as tpx
from radar.rparams_dwdxpol import RPRODSLTX
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplclr
import matplotlib.patches as mplptc
from mpl_toolkits.axes_grid1 import ImageGrid

# =============================================================================
# Define working directory, time and list files
# =============================================================================
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

rcomp2 = 'rcomp_qpe_dwd'
# rcomp = 'rcomp_qpe_dwdbxp'
# rcomp = 'rcomp_qpe_dwdjxp'
rcomp = 'rcomp_qpe_dwdxpol'
# rcomp = 'rcomp_qpe_xpol'

xlims, ylims = [4.324, 10.953], [48.635, 52.754]  # DWDXPOL RADCOV
xlims, ylims = [5.85, 11.], [48.55, 52.75]  # DWDXPOL DE

PLOT_METHODS = False

rprods_dp = ['r_adp', 'r_ah', 'r_kdp', 'r_z']
rprods_hbr = ['r_ah_kdp', 'r_kdp_zdr', 'r_z_ah', 'r_z_kdp', 'r_z_zdr']
rprods_opt = ['r_kdpopt', 'r_zopt']
rprods_hyop = ['r_ah_kdpopt', 'r_zopt_ah', 'r_zopt_kdp', 'r_zopt_kdpopt']
rprods_nmd = ['r_aho_kdpo', 'r_kdpo', 'r_zo', 'r_zo_ah', 'r_zo_kdp',
              'r_zo_zdr']

rprods = sorted(rprods_dp[1:] + rprods_hbr + rprods_opt + rprods_hyop
                + ['r_zo'])

SAVE_FIGS = True


RES_DIR = LWDIR + "pd_rdres/qpe_all/rcomp_qpe_dwd_dwdxpol/"

# RPRODSLTX = {'r_adp': '$R(A_{DP})$', 'r_ah': '$R(A_{H})$',
#              'r_kdp': '$R(K_{DP})$', 'r_z': '$R(Z_H)$',
#              'r_ah_kdp': '$R(A_{H}, K_{DP})$',
#              'r_kdp_zdr': '$R(K_{DP}, Z_{DR})$', 'r_z_ah': '$R(Z_H, A_{H})$',
#              'r_z_kdp': '$R(Z_{H}, K_{DP})$', 'r_z_zdr': '$R(Z_{H}, Z_{DR})$',
#              'r_kdpopt': '$R(K_{DP})[opt]$', 'r_zopt': '$R(Z_{H})[opt]$',
#              'r_ah_kdpopt': '$R(A_{H}, K_{DP}[opt])$',
#              'r_zopt_ah': '$R(Z_{H}[opt], A_{H})$',
#              'r_zopt_kdp': '$R(Z_{H}[opt], K_{DP})$',
#              'r_zopt_kdpopt': '$R(Z_{H}[opt], K_{DP}[opt])$',
#              'r_aho_kdpo': '$R(A_{H}, K_{DP})[evnt-spcf]$',
#              'r_kdpo': '$R(K_{DP})[evnt-spcf]$',
#              'r_zo': '$R(Z_{H})[evnt-spcf]$',
#              'r_zo_ah': '$R(Z_{H}, A_{H})[evnt-spcf]$',
#              'r_zo_kdp': '$R(Z_{H}, K_{DP})[evnt-spcf]$',
#              'r_zo_zdr': '$R(Z_{H}, Z_{DR})[evnt-spcf]$'}

# if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
RPRODSLTX['r_kdpo'] = '$R(K_{DP})[OV]$'
RPRODSLTX['r_zo'] = '$R(Z_{H})[OA]$'

# %%
# =============================================================================
# Read in Radar QPE
# =============================================================================
qpe_amlb = False
if qpe_amlb:
    appxf = '_amlb'
else:
    appxf = ''

RQPEH_DIR = (LWDIR + f'pd_rdres/qpe_all/{rcomp}/')
RQPEH_FILES = [RQPEH_DIR + i for i in sorted(os.listdir(RQPEH_DIR))
               if i.endswith('.tpy')]

daccum = []
for qpef in RQPEH_FILES:
    with open(qpef, 'rb') as fpkl:
        daccum1 = pickle.load(fpkl)
    daccum.append(daccum1)

# rqpe_acch = {}
# for rp in rprods:
#     for da in daccum:
#         'eval_rng'

eval_rqp = {rp: [] for rp in rprods}
eval_rng = {rp: [] for rp in rprods}
for rp1 in rprods:
    for da in daccum:
        eval_rqp[rp1].append(da['eval_rqp'][rp1])
        eval_rng[rp1].append(da['eval_rng'][rp1])

for k1, rp in eval_rqp.items():
    eval_rqp[k1] = np.hstack(rp)
for k1, rp in eval_rng.items():
    eval_rng[k1] = np.hstack(rp)
fres = {}
fres['altitude [m]'] = np.hstack([i['altitude [m]'] for i in daccum])

# %%
# =============================================================================
# Read in Radar QPE2
# =============================================================================
RQPEH_DIR2 = (LWDIR + f'pd_rdres/qpe_all/{rcomp2}/')
RQPEH_FILES2 = [RQPEH_DIR2 + i for i in sorted(os.listdir(RQPEH_DIR2))
                if i.endswith('.tpy')]

daccum2 = []
for qpef in RQPEH_FILES2:
    with open(qpef, 'rb') as fpkl:
        daccum1 = pickle.load(fpkl)
    daccum2.append(daccum1)

eval_rqp2 = {rp: [] for rp in rprods}
eval_rng2 = {rp: [] for rp in rprods}
for rp1 in rprods:
    for da in daccum2:
        eval_rqp2[rp1].append(da['eval_rqp'][rp1])
        eval_rng2[rp1].append(da['eval_rng'][rp1])

for k1, rp in eval_rqp2.items():
    eval_rqp2[k1] = np.hstack(rp)
for k1, rp in eval_rng2.items():
    eval_rng2[k1] = np.hstack(rp)
fres2 = {}
fres2['altitude [m]'] = np.hstack([i['altitude [m]'] for i in daccum2])

# %%

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
dtevents = [f'{x:%Y-%m-%d}' for x in START_TIMES]

all_data = [dac['eval_rng']['r_kdp'] for dac in daccum]

# Compute mean values of each dataset
means = [np.nanmean(d) for d in all_data]

# Normalise mean values for colormap
# norm = mplclr.Normalize(vmin=min(means), vmax=max(means))
norm = mplclr.Normalize(vmin=5, vmax=30)
cmap = mpl.colormaps.get_cmap('Spectral_r')

plt.style.use('default')
# plt.style.use('seaborn-v0_8')
# plt.style.use('bmh')
# plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle('Daily rainfall intensity', fontsize=16)

bplot = ax.boxplot(all_data, notch=True, showmeans=True, meanline=True,
                   sym='o', patch_artist=True, meanprops={'c': 'k'},
                   medianprops={'c': 'tab:orange'},
                   tick_labels=dtevents)

colors = [cmap(val/len(all_data)) for val in range(len(all_data))]

for patch, mean in zip(bplot['boxes'], means):
    patch.set_facecolor(cmap(norm(mean)))
# ax.violinplot(all_data, showmeans=True, showmedians=False)
# ax.set_xlabel('Rain-gauge rainfall [mm]', fontsize=14)
ax.set_ylabel('Rain-gauge rainfall [mm]', fontsize=14)
# fig.subplots_adjust(top=0.72)
ax.grid(axis='both', which='both')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.legend([bplot['medians'][0], bplot['means'][0], bplot['fliers'][0]],
          ['median', 'mean', 'outliers'], fontsize=14)
# # Add colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array(means)
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Mean Value')
plt.tight_layout()
plt.show()


# %%

cmaph = mpl.colormaps['gist_earth_r']
cmaph = mpl.colormaps['terrain']
cmaph = mpl.colormaps['berlin']
cmaph = mpl.colormaps['Spectral_r']

# lpv = {'Altitude [m]':
#        [round(np.nanmin(rg_data.ds_precip['altitude [m]']), 2),
#         round(np.nanmax(rg_data.ds_precip['altitude [m]']), -2), 25]}
lpv = {'Altitude [m]': [0, 750, 11]}
bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
       for key, value in lpv.items()}
dnorm = {'n'+key[1:]: mplclr.BoundaryNorm(
    value, cmaph.N, extend='max') for key, value in bnd.items()}

# eval_rqp = fres['eval_rqp']
# eval_rng = fres['eval_rng']

for k, v in eval_rng.items():
    v[np.isnan(eval_rqp[k])] = np.nan
    # v[eval_rqp[k] <= 1] = np.nan
    # v[v <= 1] = np.nan

for k, v in eval_rng2.items():
    v[np.isnan(eval_rqp2[k])] = np.nan
    # v[eval_rqp[k] <= 1] = np.nan
    # v[v <= 1] = np.nan

qpe_stats = {k: tpx.mstats(v, eval_rng[k]) for k, v in eval_rqp.items()}
qpe_stats2 = {k: tpx.mstats(v, eval_rng2[k]) for k, v in eval_rqp2.items()}

qpe_stats_ev = {iev['dt']: {irp: tpx.mstats(iev['eval_rqp'][irp],
                                            iev['eval_rng'][irp])
                            for irp in rprods}
                for iev in daccum}

qpe_stats_ev2 = {iev['dt']: {irp: tpx.mstats(iev['eval_rqp'][irp],
                                             iev['eval_rng'][irp])
                             for irp in rprods}
                 for iev in daccum2}

maxplt = 18
nitems = len(eval_rqp)
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
# locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
# formatter = mdates.ConciseDateFormatter(locator)
radntws = [rcomp, rcomp2]
for nrn in radntws:
    for nplot in nplots:
        fig = plt.figure(figsize=(19.2, 11))
        fig.suptitle('Daily accumulated radar'
                     + f' {nrn.replace("rcomp_qpe_", "QPE [").upper()}]'
                     + ' vs Rain-gauge measured rain totals \n', size=20)
        grid = ImageGrid(fig, 111, aspect=False,
                         nrows_ncols=(nrows, ncols), label_mode='L',
                         share_all=True, axes_pad=0.5,  cbar_location="right",
                         cbar_mode="single", cbar_size="4%", cbar_pad=0.5)
        rqpe2eval = (eval_rqp if nrn == rcomp else eval_rqp2)
        rng4eval = (eval_rng if nrn == rcomp else eval_rng2)
        fres4eval = (fres if nrn == rcomp else fres2)
        qpe_stats4eval = (qpe_stats if nrn == rcomp else qpe_stats2)
        for (axg, rprodk) in zip(grid, [k for k in sorted(rqpe2eval.keys())]):
            if rprodk in RPRODSLTX:
                rprodkltx = RPRODSLTX.get(rprodk)
            else:
                rprodkltx = rprodk
            axg.set_title(f'{rprodkltx}', size=16)
            f1 = axg.scatter(rng4eval[rprodk], rqpe2eval[rprodk], marker='o',
                             c=[fres4eval['altitude [m]']], edgecolors='k',
                             cmap=cmaph, norm=dnorm['nAltitude [m]'])
            f2 = axg.scatter(
                0, 0, marker='',
                label=(
                    f"n={qpe_stats4eval[rprodk]['N']}"
                    + f"\nr={qpe_stats4eval[rprodk]['R_Pearson [-]'][0,1]:.2f}"
                    # + f"\nMAE={qpe_stats[rprodk]['MAE']:2.2f}"
                    # + f"\nRMSE={qpe_stats[rprodk]['RMSE']:2.2f}"
                    # + f"\nNRMSE [%]={qpe_stats[rprodk]['NRMSE [%]']:2.2f}"
                    # + f"\nNMB [%]={qpe_stats[rprodk]['NMB [%]']:2.2f}"
                    # + f"\nKGE={qpe_stats[rprodk]['KGE']['kge']:2.2f}"
                    ))
            axg.axline((1, 1), slope=1, c='gray', ls='--')
            axg.set_xlabel('Rain-gauge rainfall [mm]', fontsize=17)
            axg.set_ylabel('Radar rainfall [mm]', fontsize=17)
            axg.set_xlim([0, 180])
            axg.set_ylim([0, 180])
            # axg.set_xlim([0, 80])
            # axg.set_ylim([0, 80])
            axg.grid(True)
            axg.legend(loc=2, fontsize=14, handlelength=0, handletextpad=0,
                       fancybox=True)
            # axg.xaxis.set_tick_params(labelsize=12)
            axg.tick_params(axis='both', which='major', labelsize=14)
            # axg.axes.set_aspect('equal')
            plt.show()
        axg.cax.colorbar(f1)
        axg.cax.tick_params(direction='in', which='both', labelsize=14)
        axg.cax.set_title('altitude [m]', fontsize=16)
        plt.tight_layout()
    if SAVE_FIGS:
        fname = (f"devents{len(daccum)}_{nrn}_accum_24h.png")
        plt.savefig(RES_DIR + fname, dpi=200, format='png')

# %%

theta = np.linspace(0.0, 2 * np.pi, len(qpe_stats), endpoint=False)
theta2 = np.linspace(0.0, 2 * np.pi, len(qpe_stats), endpoint=True)
theta3 = np.arange(0, (2 * np.pi), 0.0001)
theta4 = np.deg2rad(np.arange(0.0, 360, 24)+12)

lblsz = 15

# if nps == 'R_Pearson [-]' or nps == 'KGE':
#     colors = plt.get_cmap('Spectral')
# elif nps == 'NMB [%]':
#     # colors = plt.get_cmap('tpylsc_div_dbu_w_rd')
#     colors = plt.get_cmap('tpylsc_div_dbu_rd')
#     # colors = plt.get_cmap('tpylsc_div_yw_gy_bu')
#     # colors = plt.get_cmap('berlin')
# else:
#     colors = plt.get_cmap('Spectral_r')

bnd = {}
bnd['MAE'] = np.linspace(0, 15, 16)
bnd['RMSE'] = np.linspace(0, 15, 16)
bnd['NRMSE [%]'] = np.linspace(20, 100, 17)
bnd['NMB [%]'] = np.linspace(-50, 50, 11)
bnd['R_Pearson [-]'] = np.linspace(0.8, 1, 21)
bnd['KGE'] = np.linspace(0.8, 1, 21)
dnorm = {}
dnorm['MAE'] = mplclr.BoundaryNorm(bnd['MAE'], plt.get_cmap('Spectral_r').N,
                                   extend='max')
dnorm['RMSE'] = mplclr.BoundaryNorm(bnd['RMSE'], plt.get_cmap('Spectral_r').N,
                                    extend='max')
dnorm['NRMSE [%]'] = mplclr.BoundaryNorm(bnd['NRMSE [%]'],
                                         plt.get_cmap('Spectral_r').N,
                                         extend='both')
dnorm['NMB [%]'] = mplclr.BoundaryNorm(bnd['NMB [%]'],
                                       plt.get_cmap('tpylsc_div_dbu_rd').N,
                                       extend='both')
dnorm['R_Pearson [-]'] = mplclr.BoundaryNorm(bnd['R_Pearson [-]'],
                                             plt.get_cmap('Spectral').N,
                                             extend='min')
dnorm['KGE'] = mplclr.BoundaryNorm(bnd['KGE'], plt.get_cmap('Spectral').N,
                                   extend='min')


def colored_bar(left, height, z=None, width=0.5, bottom=0, ax=None, **kwargs):
    import itertools
    # import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    if ax is None:
        ax = plt.gca()
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    rects = []
    for x, y, h, w in zip(left, bottom, height, width):
        rects.append(Rectangle((x, y), w, h))
    coll = PatchCollection(rects, array=z, **kwargs)
    ax.add_collection(coll)
    ax.autoscale(enable=False, axis='y', tight=True)
    return coll


ms1 = ['D', ',', '.', '^', 's', 'P', '*', 'X', '+', 'o', '.',
       'x', '2', 'p', 'h', '|']
ms12 = ['D', ',', '.', '^', 's', 'P', '*', 'X', '+', 'X', '.',
        'x', '2', 'p', 'h', '|']

ms2 = {iev: ms1[cnt] for cnt, iev in enumerate(qpe_stats_ev)}
ms3 = {iev: ms12[cnt] for cnt, iev in enumerate(qpe_stats_ev)}

# %%
# from mpl_toolkits.axes_grid1 import make_axes_locatable
fsize = (19, 9)

stats2plot = ('R_Pearson [-]', 'KGE')

# =============================================================================
# PLOT SET OF STATS 1
# =============================================================================
fig = plt.figure(figsize=fsize)

for nps in stats2plot:
    # if stats2plot == ('R_Pearson [-]', 'KGE'):
    cmp = plt.get_cmap('Spectral')
    if nps == 'R_Pearson [-]':
        ax1 = plt.subplot(121, projection='polar')
        coll = colored_bar(theta,
                           [st[nps][0][1] for k, st in qpe_stats.items()],
                           z=[st[nps][0][1] for k, st in qpe_stats.items()],
                           width=np.radians((360/len(qpe_stats))-1), bottom=0,
                           edgecolors='k', cmap=cmp, norm=dnorm[nps])
        cb = fig.colorbar(coll, orientation='vertical', location='right',
                          fraction=0.10, shrink=0.7,
                          extend=dnorm[nps].extend, pad=.13)
        cb.ax.tick_params(labelsize=lblsz)
        coll2 = colored_bar(theta, [st[nps][0][1]
                                    for k, st in qpe_stats2.items()],
                            width=np.radians((360/len(qpe_stats2))-1),
                            bottom=0, facecolors='None', alpha=1,
                            edgecolors=('k',), label=f'{rcomp2}')
        for k1, iev in qpe_stats_ev.items():
            if k1.startswith('20210714'):
                if nps == 'R_Pearson [-]':
                    plt.scatter(
                        theta4, [[rp[nps][0][1] for k2, rp in iev.items()]],
                        c=[[rp[nps][0][1] for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k', s=64,
                        cmap=cmp, norm=dnorm[nps], label=(k1))
        for k1, iev in qpe_stats_ev2.items():
            if k1.startswith('20210714'):
                if nps == 'R_Pearson [-]':
                    plt.scatter(theta4,
                                [[rp[nps][0][1] for k2, rp in iev.items()]],
                                marker=ms3[k1], edgecolors='k', c='w', s=64,
                                label=(k1 + f'\n[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]'))
        if nps == 'R_Pearson [-]':
            plt.rgrids(np.arange(0.1, 1.01, .2), angle=0, size=lblsz,
                       fmt='%.1f', c='k')
            y = 1.19
        ax1.grid(color='gray', linestyle=':')
        ax1.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
        ax1.set_theta_offset(np.pi/2)
        ax1.set_theta_direction(-1)
        ax1.set_xticklabels([])
        ax1.set_title(nps, size=18, pad=75)
        for c1, v1 in enumerate(qpe_stats):
            x = (((np.deg2rad(360/len(qpe_stats)))/2)
                 + ((np.deg2rad(360/len(qpe_stats)))*c1))
            nrret = RPRODSLTX.get(v1)
            if '&' in nrret:
                nrret = nrret.replace('&', '$ &\n$')
            ax1.text(x, y, f'{nrret}', ha='center', va='center',
                     size=lblsz)
    fig.legend(loc='lower left', fontsize=lblsz-1)
    if nps == 'KGE':
        ax2 = plt.subplot(122, projection='polar')
        coll = colored_bar(theta,
                           [st[nps]['kge'] for k, st in qpe_stats.items()],
                           z=[st[nps]['kge'] for k, st in qpe_stats.items()],
                           width=np.radians((360/len(qpe_stats))-1), bottom=0,
                           edgecolors='grey', cmap=cmp, norm=dnorm[nps])
        cb = fig.colorbar(coll, orientation='vertical', location='right',
                          fraction=0.10, shrink=0.7,
                          extend=dnorm[nps].extend, pad=.13)
        cb.ax.tick_params(labelsize=lblsz)
        cb.ax.set_title(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
                        pad=10, size=16)
        coll2 = colored_bar(theta, [st[nps]['kge']
                                    for k, st in qpe_stats2.items()],
                            width=np.radians((360/len(qpe_stats2))-1),
                            bottom=0, facecolors='None', alpha=1,
                            edgecolors=('k'), label=f'{rcomp2}')
        for k1, iev in qpe_stats_ev.items():
            if k1.startswith('20210714'):
                if nps == 'KGE':
                    plt.scatter(
                        theta4, [[rp[nps]['kge'] for k2, rp in iev.items()]],
                        c=[[rp[nps]['kge'] for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k', s=64, cmap=cmp,
                        norm=dnorm[nps],
                        label=(k1))
        for k1, iev in qpe_stats_ev2.items():
            if k1.startswith('20210714'):
                if nps == 'KGE':
                    plt.scatter(
                        theta4, [[rp[nps]['kge'] for k2, rp in iev.items()]],
                        marker=ms3[k1], edgecolors='k', c='w', s=64,
                        label=(k1 + f'\n[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]'))
        ax2.grid(color='gray', linestyle=':')
        ax2.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
        ax2.set_theta_offset(np.pi/2)
        ax2.set_theta_direction(-1)
        ax2.set_xticklabels([])
        ax2.set_title(nps, size=18, pad=75)
        # cb.set_label(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
        #              rotation=90, size=16)
        if nps == 'KGE':
            plt.rgrids(np.arange(0.1, 1.01, .2), angle=0, size=lblsz,
                       fmt='%.1f', c='k')
            y = 1.19
        for c1, v1 in enumerate(qpe_stats):
            x = (((np.deg2rad(360/len(qpe_stats)))/2)
                 + ((np.deg2rad(360/len(qpe_stats)))*c1))
            nrret = RPRODSLTX.get(v1)
            if '&' in nrret:
                nrret = nrret.replace('&', '$ &\n$')
            ax2.text(x, y, f'{nrret}', ha='center', va='center',
                     size=lblsz)
red_patch = mplptc.Patch(
    facecolor='None', edgecolor='k',
    label=f'[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]')
fig.legend(handles=[red_patch], loc="lower right", fontsize=lblsz)
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    fnst = [ist[:ist.find('[')-1] if '[' in ist else ist for ist in stats2plot]
    fname = (f"devents{len(daccum)}_{rcomp}_{fnst[0]}_{fnst[1]}_24h.png")
    plt.savefig(RES_DIR + fname, dpi=200, format='png')

# %%
# =============================================================================
# PLOT SET OF STATS 2
# =============================================================================
stats2plot = ('RMSE', 'MAE')
# rgrids2 = np.arange(0, 13., 2)
# lpos2 = 14.28
rgrids2 = np.arange(0, 21., 5)
lpos2 = 23.8
fig = plt.figure(figsize=fsize)
for nps in stats2plot:
    # if stats2plot == ('R_Pearson [-]', 'KGE'):
    cmp = plt.get_cmap('Spectral_r')
    if nps == 'RMSE':
        ax1 = plt.subplot(121, projection='polar')
        coll = colored_bar(theta,
                           [st[nps] for k, st in qpe_stats.items()],
                           z=[st[nps] for k, st in qpe_stats.items()],
                           width=np.radians((360/len(qpe_stats))-1), bottom=0,
                           edgecolors='k', cmap=cmp, norm=dnorm[nps])
        cb = fig.colorbar(coll, orientation='vertical', location='right',
                          fraction=0.10, shrink=0.7, extend=dnorm[nps].extend,
                          format=mpl.ticker.FormatStrFormatter('%.1f'),
                          pad=.13)
        coll2 = colored_bar(theta, [st[nps]
                                    for k, st in qpe_stats2.items()],
                            width=np.radians((360/len(qpe_stats2))-1),
                            bottom=0, facecolors='None', alpha=1,
                            edgecolors=('k',), label=f'{rcomp2}')
        for k1, iev in qpe_stats_ev.items():
            if k1.startswith('20210714'):
                if nps == 'RMSE':
                    plt.scatter(
                        theta4, [[rp[nps] for k2, rp in iev.items()]],
                        c=[[rp[nps] for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k', s=64,
                        cmap=cmp, norm=dnorm[nps], label=(k1))
        for k1, iev in qpe_stats_ev2.items():
            if k1.startswith('20210714'):
                if nps == 'RMSE':
                    plt.scatter(theta4,
                                [[rp[nps] for k2, rp in iev.items()]],
                                marker=ms3[k1], edgecolors='k', c='w', s=64,
                                label=(k1 + f'\n[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]'))
        if nps == 'RMSE':
            # plt.rgrids(np.arange(0, 21., 5), angle=0, size=lblsz, fmt='%.0f',
            #            c='k')
            # y = 23.8
            plt.rgrids(rgrids2, angle=0, size=lblsz, fmt='%.0f', c='k')
            # y = lpos2
        ax1.grid(color='gray', linestyle=':')
        ax1.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
        ax1.set_theta_offset(np.pi/2)
        ax1.set_theta_direction(-1)
        ax1.set_xticklabels([])
        ax1.set_title(nps, size=18, pad=75)
        # fig.delaxes(fig.axes[1])
        cb.ax.tick_params(labelsize=lblsz)
        for c1, v1 in enumerate(qpe_stats):
            x = (((np.deg2rad(360/len(qpe_stats)))/2)
                 + ((np.deg2rad(360/len(qpe_stats)))*c1))
            # ax1.text(x, lpos2, f'{RPRODSLTX.get(v1)}', ha='center',
            #          va='center', size=lblsz)
            nrret = RPRODSLTX.get(v1)
            if '&' in nrret:
                nrret = nrret.replace('&', '$ &\n$')
            ax1.text(x, lpos2, f'{nrret}', ha='center', va='center',
                     size=lblsz)
    fig.legend(loc='lower left', fontsize=lblsz-1)
    if nps == 'MAE':
        ax2 = plt.subplot(122, projection='polar')
        coll = colored_bar(theta,
                           [st[nps] for k, st in qpe_stats.items()],
                           z=[st[nps] for k, st in qpe_stats.items()],
                           width=np.radians((360/len(qpe_stats))-1), bottom=0,
                           edgecolors='grey', cmap=cmp, norm=dnorm[nps])
        cb = fig.colorbar(coll, orientation='vertical', location='right',
                          fraction=0.10, shrink=0.7, extend=dnorm[nps].extend,
                          format=mpl.ticker.FormatStrFormatter('%.1f'),
                          pad=.13)
        coll2 = colored_bar(theta, [st[nps] for k, st in qpe_stats2.items()],
                            width=np.radians((360/len(qpe_stats2))-1),
                            bottom=0, facecolors='None', alpha=1,
                            edgecolors=('k'), label=f'{rcomp2}')
        for k1, iev in qpe_stats_ev.items():
            if k1.startswith('20210714'):
                if nps == 'MAE':
                    plt.scatter(
                        theta4, [[rp[nps] for k2, rp in iev.items()]],
                        c=[[rp[nps] for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k', s=64, cmap=cmp,
                        norm=dnorm[nps],
                        label=(k1))
        for k1, iev in qpe_stats_ev2.items():
            if k1.startswith('20210714'):
                if nps == 'MAE':
                    plt.scatter(
                        theta4, [[rp[nps] for k2, rp in iev.items()]],
                        marker=ms3[k1], edgecolors='k', c='w', s=64,
                        label=(k1 + f'\n[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]'))
        ax2.grid(color='gray', linestyle=':')
        ax2.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
        ax2.set_theta_offset(np.pi/2)
        ax2.set_theta_direction(-1)
        ax2.set_xticklabels([])
        ax2.set_title(nps, size=18, pad=75)
        cb.ax.tick_params(labelsize=lblsz)
        cb.ax.set_title(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
                        pad=10, size=16)
        # cb.set_label(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
        #              rotation=90, size=16)
        if nps == 'MAE':
            # plt.rgrids(np.arange(0, 21., 5), angle=0, size=lblsz, fmt='%.0f',
            #            c='k')
            # y = 23.8
            plt.rgrids(rgrids2, angle=0, size=lblsz, fmt='%.0f',
                       c='k')
        for c1, v1 in enumerate(qpe_stats):
            x = (((np.deg2rad(360/len(qpe_stats)))/2)
                 + ((np.deg2rad(360/len(qpe_stats)))*c1))
            # ax2.text(x, lpos2, f'{RPRODSLTX.get(v1)}', ha='center',
            #          va='center', size=lblsz)
            nrret = RPRODSLTX.get(v1)
            if '&' in nrret:
                nrret = nrret.replace('&', '$ &\n$')
            ax2.text(x, lpos2, f'{nrret}', ha='center', va='center',
                     size=lblsz)
red_patch = mplptc.Patch(
    facecolor='None', edgecolor='k',
    label=f'[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]')
fig.legend(handles=[red_patch], loc="lower right", fontsize=lblsz)
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    fnst = [ist[:ist.find('[')-1] if '[' in ist else ist for ist in stats2plot]
    fname = (f"devents{len(daccum)}_{rcomp}_{fnst[0]}_{fnst[1]}_24h.png")
    plt.savefig(RES_DIR + fname, dpi=200, format='png')

# %%
# =============================================================================
# PLOT SET OF STATS 3
# =============================================================================
stats2plot = ('NRMSE [%]', 'NMB [%]')
fig = plt.figure(figsize=fsize)
for nps in stats2plot:
    cmp = plt.get_cmap('Spectral_r')
    if nps == 'NRMSE [%]':
        ax1 = plt.subplot(121, projection='polar')
        coll = colored_bar(theta, [st[nps] for k, st in qpe_stats.items()],
                           z=[st[nps] for k, st in qpe_stats.items()],
                           width=np.radians((360/len(qpe_stats))-1), bottom=0,
                           edgecolors='k', cmap=cmp, norm=dnorm[nps])
        cb = fig.colorbar(coll, orientation='vertical', location='right',
                          fraction=0.10, shrink=0.7, extend=dnorm[nps].extend,
                          format=mpl.ticker.FormatStrFormatter('%.0f'),
                          pad=.13)
        cb.ax.tick_params(labelsize=lblsz)
        cb.ax.set_title(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
                        pad=10, size=16)
        # fig.delaxes(fig.axes[1])
        coll2 = colored_bar(theta, [st[nps] for k, st in qpe_stats2.items()],
                            width=np.radians((360/len(qpe_stats2))-1),
                            bottom=0, facecolors='None', alpha=1,
                            edgecolors=('k',), label=f'{rcomp2}')
        for k1, iev in qpe_stats_ev.items():
            if k1.startswith('20210714'):
                if nps == 'NRMSE [%]':
                    plt.scatter(
                        theta4, [[rp[nps] for k2, rp in iev.items()]],
                        c=[[rp[nps] for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k', s=64,
                        cmap=cmp, norm=dnorm[nps], label=(k1))
        for k1, iev in qpe_stats_ev2.items():
            if k1.startswith('20210714'):
                if nps == 'NRMSE [%]':
                    plt.scatter(theta4,
                                [[rp[nps] for k2, rp in iev.items()]],
                                marker=ms3[k1], edgecolors='k', c='w', s=64,
                                label=(k1 + f'\n[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]'))
        if nps == 'NRMSE [%]':
            plt.rgrids(np.arange(0, 110, 20), angle=0, size=lblsz, fmt='%.0f',
                       c='k')
            y = 119
        ax1.grid(color='gray', linestyle=':')
        ax1.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
        ax1.set_theta_offset(np.pi/2)
        ax1.set_theta_direction(-1)
        ax1.set_xticklabels([])
        ax1.set_title(nps, size=18, pad=75)
        for c1, v1 in enumerate(qpe_stats):
            x = (((np.deg2rad(360/len(qpe_stats)))/2)
                 + ((np.deg2rad(360/len(qpe_stats)))*c1))
            # ax1.text(x, y, f'{RPRODSLTX.get(v1)}', ha='center', va='center',
            #          size=lblsz)
            nrret = RPRODSLTX.get(v1)
            if '&' in nrret:
                nrret = nrret.replace('&', '$ &\n$')
            ax1.text(x, y, f'{nrret}', ha='center', va='center',
                     size=lblsz)
    fig.legend(loc='lower left', fontsize=lblsz-1)
    if nps == 'NMB [%]':
        cmp = plt.get_cmap('tpylsc_div_dbu_rd')
        ax2 = plt.subplot(122, projection='polar')
        coll = colored_bar(theta, [st[nps] for k, st in qpe_stats.items()],
                           z=[st[nps] for k, st in qpe_stats.items()],
                           width=np.radians((360/len(qpe_stats))-1), bottom=0,
                           edgecolors='k', cmap=cmp, norm=dnorm[nps])
        cb = fig.colorbar(coll, orientation='vertical', location='right',
                          fraction=0.10, shrink=0.7, extend=dnorm[nps].extend,
                          format=mpl.ticker.FormatStrFormatter('%.0f'),
                          pad=.13)
        cb.ax.tick_params(labelsize=lblsz)
        cb.ax.set_title(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
                        pad=10, size=16)
        coll2 = colored_bar(theta, [st[nps] for k, st in qpe_stats2.items()],
                            width=np.radians((360/len(qpe_stats2))-1),
                            bottom=0, facecolors='None', alpha=1,
                            edgecolors=('k',), label=f'{rcomp2}')
        for k1, iev in qpe_stats_ev.items():
            if k1.startswith('20210714'):
                if nps == 'NMB [%]':
                    plt.scatter(
                        theta4, [[rp[nps] for k2, rp in iev.items()]],
                        c=[[rp[nps] for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k', s=64, cmap=cmp,
                        norm=dnorm[nps],
                        label=(k1))
        for k1, iev in qpe_stats_ev2.items():
            if k1.startswith('20210714'):
                if nps == 'NMB [%]':
                    plt.scatter(
                        theta4, [[rp[nps] for k2, rp in iev.items()]],
                        marker=ms3[k1], edgecolors='k', c='w', s=64,
                        label=(k1 + f'\n[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]'))
        ax2.grid(color='gray', linestyle=':')
        ax2.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
        ax2.set_theta_offset(np.pi/2)
        ax2.set_theta_direction(-1)
        ax2.set_xticklabels([])
        ax2.set_title(nps, size=18, pad=75)
        # cb.set_label(f'\n[{rcomp.replace("rcomp_qpe", "QPE").upper()}]',
        #              rotation=90, size=16)
        if nps == 'NMB [%]':
            plt.rgrids(np.arange(-50, 60, 25), angle=0, size=lblsz, fmt='%.0f',
                       c='k')
            y = 69
            plt.polar(theta3, np.zeros_like(theta3), 'k--')
        for c1, v1 in enumerate(qpe_stats):
            x = (((np.deg2rad(360/len(qpe_stats)))/2)
                 + ((np.deg2rad(360/len(qpe_stats)))*c1))
            # ax2.text(x, y, f'{RPRODSLTX.get(v1)}', ha='center', va='center',
            #          size=lblsz)
            nrret = RPRODSLTX.get(v1)
            if '&' in nrret:
                nrret = nrret.replace('&', '$ &\n$')
            ax2.text(x, y, f'{nrret}', ha='center', va='center',
                     size=lblsz)
red_patch = mplptc.Patch(
    facecolor='None', edgecolor='k',
    label=f'[{rcomp2.replace("rcomp_qpe", "QPE").upper()}]')
fig.legend(handles=[red_patch], loc="lower right", fontsize=lblsz)
# plt.subplots_adjust(top=0.945, bottom=0.017, left=0.061, right=0.985,
#                     wspace=0.200, hspace=0.159)
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    fnst = [ist[:ist.find('[')-1] if '[' in ist else ist for ist in stats2plot]
    fname = (f"devents{len(daccum)}_{rcomp}_{fnst[0]}_{fnst[1]}_24h.png")
    plt.savefig(RES_DIR + fname, dpi=200, format='png')

# if SAVE_FIGS:
#     if '[' in stat2plot:
#         fnst = stat2plot[:stat2plot.find('[')-1]
#     else:
#         fnst = stat2plot
#     fname = (f"devents{len(daccum)}_{rcomp}_{fnst}_24h.png")
#     plt.savefig(RES_DIR + fname, format='png')

