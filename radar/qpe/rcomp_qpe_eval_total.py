#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:51:02 2025

@author: dsanchez
"""

import os
import pickle
import numpy as np
from radar import twpext as tpx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from mpl_toolkits.axes_grid1 import ImageGrid

# =============================================================================
# Define working directory, time and list files
# =============================================================================
LWDIR = '/home/dsanchez/sciebo_dsr/'
EWDIR = '/run/media/dsanchez/PSDD1TB/safe/bonn_postdoc/'
# LWDIR = '/home/enchiladaszen/Documents/sciebo/'
# EWDIR = '/media/enchiladaszen/PSDD1TB/safe/bonn_postdoc/'

rcomp = 'rcomp_qpe_dwd'
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
# SAVE_DATA = False

RES_DIR = LWDIR + f"pd_rdres/qpe_all/{rcomp}/"

rprodsltx = {'r_adp': '$R(A_{DP})$', 'r_ah': '$R(A_{H})$',
             'r_kdp': '$R(K_{DP})$', 'r_z': '$R(Z_H)$',
             'r_ah_kdp': '$R(A_{H}, K_{DP})$',
             'r_kdp_zdr': '$R(K_{DP}, Z_{DR})$', 'r_z_ah': '$R(Z_H, A_{H})$',
             'r_z_kdp': '$R(Z_{H}, K_{DP})$', 'r_z_zdr': '$R(Z_{H}, Z_{DR})$',
             'r_kdpopt': '$R(K_{DP})[opt]$', 'r_zopt': '$R(Z_{H})[opt]$',
             'r_ah_kdpopt': '$R(A_{H}, K_{DP}[opt])$',
             'r_zopt_ah': '$R(Z_{H}[opt], A_{H})$',
             'r_zopt_kdp': '$R(Z_{H}[opt], K_{DP})$',
             'r_zopt_kdpopt': '$R(Z_{H}[opt], K_{DP}[opt])$',
             'r_aho_kdpo': '$R(A_{H}, K_{DP})[evnt-spcf]$',
             'r_kdpo': '$R(K_{DP})[evnt-spcf]$',
             'r_zo': '$R(Z_{H})[evnt-spcf]$',
             'r_zo_ah': '$R(Z_{H}, A_{H})[evnt-spcf]$',
             'r_zo_kdp': '$R(Z_{H}, K_{DP})[evnt-spcf]$',
             'r_zo_zdr': '$R(Z_{H}, Z_{DR})[evnt-spcf]$'}

# if START_TIME != dt.datetime(2021, 7, 14, 0, 0):
rprodsltx['r_kdpo'] = '$R(K_{DP})[OV]$'
rprodsltx['r_zo'] = '$R(Z_{H})[OA]$'



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
cmaph = mpl.colormaps['gist_earth_r']
cmaph = mpl.colormaps['Spectral_r']
# lpv = {'Altitude [m]':
#        [round(np.nanmin(rg_data.ds_precip['altitude [m]']), 2),
#         round(np.nanmax(rg_data.ds_precip['altitude [m]']), -2), 25]}
lpv = {'Altitude [m]':
       [0, 750, 11]}
bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
       for key, value in lpv.items()}
dnorm = {'n'+key[1:]: mpc.BoundaryNorm(
    value, cmaph.N, extend='max') for key, value in bnd.items()}

# eval_rqp = fres['eval_rqp']

# eval_rng = fres['eval_rng']

for k, v in eval_rng.items():
    v[np.isnan(eval_rqp[k])] = np.nan
    # v[eval_rqp[k] <= 1] = np.nan
    # v[v <= 1] = np.nan

qpe_stats = {k: tpx.mstats(v, eval_rng[k]) for k, v in eval_rqp.items()}
qpe_stats2 = {iev['dt']: {irp: tpx.mstats(iev['eval_rqp'][irp],
                                          iev['eval_rng'][irp])
                          for irp in rprods}
              for iev in daccum}


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
for nplot in nplots:
    fig = plt.figure(figsize=(19.2, 11))
    fig.suptitle('Daily accumulated radar QPE vs Rain-gauge measured'
                 ' rain totals \n', size=16)
    grid = ImageGrid(fig, 111, aspect=False,
                     nrows_ncols=(nrows, ncols), label_mode='L',
                     share_all=True, axes_pad=0.5,  cbar_location="right",
                     cbar_mode="single", cbar_size="4%", cbar_pad=0.5)
    for (axg, rprodk) in zip(grid, [k for k in sorted(eval_rqp.keys())]):
        if rprodk in rprodsltx:
            rprodkltx = rprodsltx.get(rprodk)
        else:
            rprodkltx = rprodk
        axg.set_title(f'{rprodkltx}', size=14)
        f1 = axg.scatter(eval_rng[rprodk], eval_rqp[rprodk], marker='o',
                         c=[fres['altitude [m]']],
                         edgecolors='k',
                         cmap=cmaph, norm=dnorm['nAltitude [m]'])
        f2 = axg.scatter(
            0, 0, marker='',
            label=(
                f"n={qpe_stats[rprodk]['N']}"
                + f"\nr={qpe_stats[rprodk]['R_Pearson [-]'][0,1]:.2f}"
                # + f"\nMAE={qpe_stats[rprodk]['MAE']:2.2f}"
                # + f"\nRMSE={qpe_stats[rprodk]['RMSE']:2.2f}"
                # + f"\nNRMSE [%]={qpe_stats[rprodk]['NRMSE [%]']:2.2f}"
                # + f"\nNMB [%]={qpe_stats[rprodk]['NMB [%]']:2.2f}"
                # + f"\nKGE={qpe_stats[rprodk]['KGE']['kge']:2.2f}"
                ))
        axg.axline((1, 1), slope=1, c='gray', ls='--')
        axg.set_xlabel('Rain-gauge rainfall [mm]', fontsize=14)
        axg.set_ylabel('Radar rainfall [mm]', fontsize=14)
        axg.set_xlim([0, 180])
        axg.set_ylim([0, 180])
        # axg.set_xlim([0, 80])
        # axg.set_ylim([0, 80])
        axg.grid(True)
        axg.legend(loc=2, fontsize=12, handlelength=0, handletextpad=0,
                   fancybox=True)
        # axg.xaxis.set_tick_params(labelsize=12)
        axg.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
    # nitems = len(eval_rqp)
    axg.cax.colorbar(f1)
    axg.cax.tick_params(direction='in', which='both', labelsize=14)
    # axg.cax.toggle_label(True)
    axg.cax.set_title('altitude [m]', fontsize=14)
    plt.tight_layout()

if SAVE_FIGS:
    fname = (f"devents{len(daccum)}_{rcomp}_accum_24h.png")
    plt.savefig(RES_DIR + fname, format='png')

# %%

theta = np.linspace(0.0, 2 * np.pi, len(qpe_stats), endpoint=False)
theta2 = np.linspace(0.0, 2 * np.pi, len(qpe_stats), endpoint=True)
theta3 = np.arange(0, (2 * np.pi), 0.0001)
theta4 = np.deg2rad(np.arange(0.0, 360, 24)+12)

stat2plot = 'MAE'
stat2plot = 'RMSE'
stat2plot = 'NRMSE [%]'
stat2plot = 'NMB [%]'
stat2plot = 'R_Pearson [-]'
stat2plot = 'KGE'

if stat2plot == 'R_Pearson [-]':
    colors = plt.get_cmap('Spectral')
elif stat2plot == 'NMB [%]':
    # colors = plt.get_cmap('tpylsc_div_dbu_w_rd')
    colors = plt.get_cmap('tpylsc_div_dbu_rd')
    # colors = plt.get_cmap('tpylsc_div_yw_gy_bu')
    # colors = plt.get_cmap('berlin')
else:
    colors = plt.get_cmap('Spectral_r')


bnd = {}
bnd['MAE'] = np.linspace(0, 15, 16)
bnd['RMSE'] = np.linspace(0, 15, 16)
bnd['NRMSE [%]'] = np.linspace(20, 100, 17)
bnd['NMB [%]'] = np.linspace(-50, 50, 11)
bnd['R_Pearson [-]'] = np.linspace(0.8, 1, 11)
bnd['KGE'] = np.linspace(0., 1, 11)
dnorm = {}
dnorm['MAE'] = mpc.BoundaryNorm(bnd['MAE'], colors.N, extend='max')
dnorm['RMSE'] = mpc.BoundaryNorm(bnd['RMSE'], colors.N, extend='max')
dnorm['NRMSE [%]'] = mpc.BoundaryNorm(bnd['NRMSE [%]'], colors.N,
                                      extend='both')
dnorm['NMB [%]'] = mpc.BoundaryNorm(bnd['NMB [%]'], colors.N, extend='both')
dnorm['R_Pearson [-]'] = mpc.BoundaryNorm(bnd['R_Pearson [-]'],
                                          colors.N, extend='min')
dnorm['KGE'] = mpc.BoundaryNorm(bnd['KGE'], colors.N, extend='max')


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

# ls1 = ['-', ':', '--', '-.', (0, (1, 1)), (5, (10, 3)),  (0, (5, 1)),
#        (0, (3, 1, 1, 1)),
#        (0, (3, 5, 1, 5, 1, 5)),
#        (0, (3, 10, 1, 10, 1, 10)),
#        (0, (3, 1, 1, 1, 1, 1))]


ms1 = ['D', ',', '.', '^', 's', 'P', '*', 'X', '+', 'o', '.',
       'x', '2', 'p', 'h', '|']

ms2 = {iev: ms1[cnt] for cnt, iev in enumerate(qpe_stats2)}

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='polar')
if stat2plot == 'R_Pearson [-]':
    coll = colored_bar(theta,
                       [st[stat2plot][0][1] for k, st in qpe_stats.items()],
                       z=[st[stat2plot][0][1] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       edgecolors='k', cmap=colors, norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend='min', pad=.1)
elif stat2plot == 'KGE':
    coll = colored_bar(theta,
                       [st[stat2plot]['kge'] for k, st in qpe_stats.items()],
                       z=[st[stat2plot]['kge'] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       edgecolors='k', cmap=colors, norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend='max', pad=.1)
elif stat2plot == 'MAE' or stat2plot == 'RMSE':
    coll = colored_bar(theta, [st[stat2plot] for k, st in qpe_stats.items()],
                       z=[st[stat2plot] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       edgecolors='k', cmap=colors,
                       norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend='max', pad=.1)
    # coll2 = colored_bar(theta, [st[stat2plot] for k, st in qpe_stats.items()],
    #                     width=np.radians((360/len(qpe_stats))-1), bottom=0,
    #                     facecolors='None',alpha=0.5,  edgecolors=("black",))
else:
    coll = colored_bar(theta, [st[stat2plot] for k, st in qpe_stats.items()],
                       z=[st[stat2plot] for k, st in qpe_stats.items()],
                       width=np.radians((360/len(qpe_stats))-1), bottom=0,
                       edgecolors='k', cmap=colors, norm=dnorm[stat2plot])
    cb = fig.colorbar(coll, orientation='horizontal', location='top',
                      fraction=0.10, shrink=0.8,
                      extend='both', pad=.1)
    if stat2plot == 'NMB [%]':
        plt.polar(theta3,
                  np.zeros_like(theta3), 'k--')
for k1, iev in qpe_stats2.items():
    if k1.startswith('20210714'):
        if stat2plot == 'R_Pearson [-]':
            plt.scatter(theta4, [[rp[stat2plot][0][1]
                                 for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k',
                        c=[[rp[stat2plot][0][1] for k2, rp in iev.items()]],
                        cmap=colors, norm=dnorm[stat2plot], label=k1)
        elif stat2plot == 'KGE':
            plt.scatter(theta4, [[rp[stat2plot]['kge']
                                 for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k',
                        c=[[rp[stat2plot]['kge'] for k2, rp in iev.items()]],
                        cmap=colors, norm=dnorm[stat2plot], label=k1)
        else:
            plt.scatter(theta4, [[rp[stat2plot]
                                 for k2, rp in iev.items()]],
                        marker=ms2[k1], edgecolors='k',
                        c=[[rp[stat2plot] for k2, rp in iev.items()]],
                        cmap=colors, norm=dnorm[stat2plot], label=k1)
cb.ax.set_title(stat2plot, size=18)
cb.ax.tick_params(labelsize=14)
ax.grid(color='gray', linestyle=':')
ax.set_thetagrids(np.arange(0, 360, 360/len(qpe_stats)))
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticklabels([])
if stat2plot == 'R_Pearson [-]':
    plt.rgrids(np.arange(0.1, 1.01, .2), angle=0, size=14, fmt='%.1f', c='k')
    y = 1.19
elif stat2plot == 'KGE':
    plt.rgrids(np.arange(0.1, 1.01, .2), angle=0, size=14, fmt='%.1f', c='k')
    y = 1.19
elif stat2plot == 'NRMSE [%]':
    plt.rgrids(np.arange(0, 110, 20), angle=0, size=14, fmt='%.0f', c='k')
    y = 119
elif stat2plot == 'NMB [%]':
    plt.rgrids(np.arange(-50, 60, 25), angle=0, size=14, fmt='%.1f', c='k')
    y = 69
elif stat2plot == 'RMSE':
    plt.rgrids(np.arange(0, 21., 5), angle=0, size=14, fmt='%.0f', c='k')
    y = 23.8
else:
    plt.rgrids(np.arange(0, 21., 5), angle=0, size=14, fmt='%.0f', c='k')
    y = 23.8
# plt.rgrids(np.arange(0, 16., 5), angle=90, size=10, fmt='%.2f')
for c1, v1 in enumerate(qpe_stats):
    x = (((np.deg2rad(360/len(qpe_stats)))/2)
         + ((np.deg2rad(360/len(qpe_stats)))*c1))
    ax.text(x, y, f'{rprodsltx.get(v1)}', ha='center', va='center',
            size=14)
# pos = ax.get_rlabel_position()
# ax.set_rlabel_position(pos+157.5)
plt.tight_layout()
fig.legend(loc="lower left")
# ax.set_title(f'bar length:''\n''RMSE [km]', fontsize=28, x=0, y=-.1)
# ax.set_title(f'bar length: \n {stat2plot}', fontsize=12, x=0, y=-.08)
plt.show()

if SAVE_FIGS:
    if '[' in stat2plot:
        fnst = stat2plot[:stat2plot.find('[')-1]
    else:
        fnst = stat2plot
    fname = (f"devents{len(daccum)}_{rcomp}_{fnst}_24h.png")
    plt.savefig(RES_DIR + fname, format='png')
